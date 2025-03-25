#用来计算原模型meditron-7B在medmcqa上的准确率
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn

from datasets import load_from_disk

# 1. 加载dev数据集
processed_data = load_from_disk("/root/meditron-medmcqa-finetune/data/processed_dataset")
dev_dataset = processed_data["dev"]



# 2. 自定义collate_fn
def dev_collate_fn(batch):
    for sample in batch:
        if sample is not None:
            # 将 None 替换为默认值，例如空字符串
            if sample.get("topic_name") is None:
                sample["topic_name"] = ""
            if sample.get("exp") is None:
                sample["exp"] = ""
    prompts = [sample["prompt"] for sample in batch]
    gold_labels = [sample["label"] for sample in batch]
    return {"prompts": prompts, "gold_labels": gold_labels}


# 3. 构造 DataLoader
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)


# 4. 计算每个文本的loss（与原代码相同）
def compute_per_example_loss(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    labels = inputs["input_ids"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    B, L, V = shift_logits.shape
    loss_tokens = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
    loss_tokens = loss_tokens.view(B, L)

    attention_mask = (shift_labels != tokenizer.pad_token_id).float()
    per_example_loss = (loss_tokens * attention_mask).sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8)
    return per_example_loss


# 5. 评估函数（与原代码相同）
def evaluate_accuracy_batch(model, tokenizer, dev_loader, print_accu_interval=500):
    options = ["A", "B", "C", "D"]
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            prompts = batch["prompts"]
            gold_labels = batch["gold_labels"]
            batch_size = len(prompts)

            # 为每个 prompt 生成4个候选文本
            candidate_texts = []
            for prompt in prompts:
                for opt in options:
                    candidate_texts.append(prompt + " " + opt)

            losses = compute_per_example_loss(model, tokenizer, candidate_texts)
            losses = losses.view(batch_size, 4)

            pred_indices = torch.argmin(losses, dim=1)
            pred_labels = [options[idx] for idx in pred_indices.cpu().numpy()]

            for pred, gold in zip(pred_labels, gold_labels):
                if pred == gold:
                    correct += 1
                total += 1
                if total % print_accu_interval == 0:
                    interim_accuracy = correct / total
                    print(f"Processed {total} samples, Interim Accuracy: {interim_accuracy * 100:.2f}%")

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 6. 加载“未微调”的原始 Meditron-7B
base_model_path = "/root/meditron-medmcqa-finetune/models/meditron-7b"
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
model.to("cuda")

# 7. 加载 tokenizer（不要加载 LoRA 目录，使用原模型或单独保存的 tokenizer）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token  # 保持与训练一致

# 8. 计算验证集准确率
print_accu_interval = 200
accuracy = evaluate_accuracy_batch(model, tokenizer, dev_loader, print_accu_interval)
print(f"Validation Accuracy without LoRA: {accuracy * 100:.2f}%")
