import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #启用 PyTorch 的更智能显存分配策略

processed_data = load_from_disk("home/ubuntu/meditron-medmcqa-finetune/data/processed_dataset")
dev_dataset = processed_data["dev"]
dev_subset = dev_dataset.shuffle(seed=42).select(range(1000)) #先用验证集的一部分进行计算

# 1. 自定义 collate_fn
def dev_collate_fn(batch):
    for sample in batch:
        if sample is not None:
            # 将 None 替换为默认值，例如空字符串
            if sample.get("topic_name") is None:
                sample["topic_name"] = ""
            if sample.get("exp") is None:
                sample["exp"] = ""
    # 将每个样本的 prompt 和 label 分别收集到列表中
    prompts = [sample["prompt"] for sample in batch]
    gold_labels = [sample["label"] for sample in batch]
    return {"prompts": prompts, "gold_labels": gold_labels}

# 2. 构造 DataLoader（可以根据你的硬件适当调整 batch_size）
dev_loader = DataLoader(dev_subset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)

# 3. 定义一个函数，批量计算每个文本的 per-example loss
def compute_per_example_loss(model, tokenizer, texts):
    """
    texts: list of string，每个文本为 "prompt + ' ' + option"
    返回一个 tensor，形状为 (len(texts),)，每个元素为对应文本的 loss
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 前向传播得到 logits
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits  # shape: (B, L, vocab_size)
    labels = inputs["input_ids"]  # shape: (B, L)

    # 为因果语言模型做 shift（注意：shift后两者长度对齐）
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 计算每个 token 的 loss，不作 reduction
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    B, L, V = shift_logits.shape
    loss_tokens = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
    loss_tokens = loss_tokens.view(B, L)

    # 创建 attention mask：非 pad 的位置为 1
    attention_mask = (shift_labels != tokenizer.pad_token_id).float()
    # 每个样本的 loss：对有效 token 的 loss 求和后归一化
    per_example_loss = (loss_tokens * attention_mask).sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8)
    return per_example_loss  # shape: (B,)

# 4. 定义评估函数，利用批量评估计算准确率
def evaluate_accuracy_batch(model, tokenizer, dev_loader, print_accu_interval = 500):
    options = ["A", "B", "C", "D"]
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            prompts = batch["prompts"]        # list of str, 长度为 batch_size
            gold_labels = batch["gold_labels"]  # list of str, 长度为 batch_size
            batch_size = len(prompts)

            # 对每个样本生成 4 个候选文本
            candidate_texts = []
            for prompt in prompts:
                for opt in options:
                    candidate_texts.append(prompt + " " + opt)

            # candidate_texts 长度为 batch_size * 4
            losses = compute_per_example_loss(model, tokenizer, candidate_texts)
            # losses 形状: (batch_size * 4,)，重塑为 (batch_size, 4)
            losses = losses.view(batch_size, 4)

            # 每个样本选择 loss 最低的选项作为预测
            pred_indices = torch.argmin(losses, dim=1)  # shape: (batch_size,)
            pred_labels = [options[idx] for idx in pred_indices.cpu().numpy()]

            for pred, gold in zip(pred_labels, gold_labels):
                if pred == gold:
                    correct += 1
                total += 1
                # 每累计500个样本输出一次当前准确率
                if total % print_accu_interval == 0:
                    interim_accuracy = correct / total
                    print(f"Processed {total} samples, Interim Accuracy: {interim_accuracy * 100:.2f}%")
    accuracy = correct / total if total > 0 else 0
    return accuracy

# 5. 加载保存好的 LoRA 模型和 tokenizer（示例路径，根据实际调整）
base_model_path = "/root/meditron-medmcqa-finetune/models/meditron-7b"
lora_checkpoint_path = "/root/meditron-medmcqa-finetune/data/train_4/epoch_3"

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("/root/meditron-medmcqa-finetune/data/train_4/tokenizer")
tokenizer.pad_token = tokenizer.eos_token #和训练时保持一致

# 6. 计算验证集准确率
print_accu_interval = 200
accuracy = evaluate_accuracy_batch(model, tokenizer, dev_loader, print_accu_interval)
print(f"Validation Accuracy: {accuracy*100:.2f}%")