from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
from peft import PeftModel
from datasets import load_from_disk
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
import torch.nn as nn
import csv
from pathlib import Path

from src.train_12 import save_path


def train_model(lora_rank=8, dropout=0.1, learning_rate=1e-4):


    # 环境变量优化 CUDA 显存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ✅ 用 pathlib 指定本地模型路径
    base_dir = Path("/home/ubuntu/meditron-medmcqa-finetune")  # 修改为你的项目根目录
    model_path = base_dir / "models" / "meditron-7b"

    # 1) 加载 Tokenizer（本地）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token  # 避免出现警告

    # 2) 8-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # 3) 加载模型（本地）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True  # 加这句！
    )


    #4) 加载数据
    processed_data = load_from_disk("/home/ubuntu/meditron-medmcqa-finetune/data/processed_dataset_deepseek")
    # 里面包含 train/dev/test 分割
    train_dataset = processed_data["train"]
    dev_dataset = processed_data["dev"]

    def format_example(example):
        # 如果你之前在 "prompt" 字段已经包含了 "Answer: ???"
        # 并且 "label" 是 "A/B/C/D"
        # 这里直接把它拼到 prompt 后面即可
        text = example["prompt"] + " " + example["label"]  # 例如: "...Answer: C"
        return {"input_text": text}

    # 5) 构建Lora模型
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2*lora_rank,
        lora_dropout=dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to("cuda")  # 确保模型在GPU上

    # 6) 数据加载器
    train_dataset = train_dataset.map(format_example)
    dev_dataset = dev_dataset.map(format_example)
    train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
    dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
    train_subset = train_dataset.shuffle(seed=42).select(range(10000))
    dev_subset = dev_dataset.shuffle(seed=42).select(range(1000))  # 先用验证集的一部分进行计算


    def my_collate_fn(batch):
        for sample in batch:
            if sample is not None:
                # 将 None 替换为默认值，例如空字符串
                if sample.get("topic_name") is None:
                    sample["topic_name"] = ""
                if sample.get("exp") is None:
                    sample["exp"] = ""
        # 过滤掉整体为 None 的样本
        filtered_batch = [sample for sample in batch if sample is not None]
        if len(filtered_batch) == 0:
            raise ValueError("过滤后，当前批次没有有效样本，请检查数据预处理逻辑")
        return torch.utils.data.dataloader.default_collate(filtered_batch)

    train_dataloader = DataLoader(train_subset, batch_size=3, shuffle=True, collate_fn=my_collate_fn)
    dev_subset_dataloader = DataLoader(train_subset, batch_size=3, shuffle=True, collate_fn=my_collate_fn)
    # 7) 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 8) 训练循环
    epochs = 3
    accumulation_steps = 5  # 每5个mini-batch累积一次梯度
    global_step = 0  # 记录真实的优化步数（每完成一次optimizer.step()就+1）

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # 确保每个epoch开始时梯度为0

        for i, batch in enumerate(train_dataloader):
            # 1. 准备输入
            if "input_text" in batch:
                inputs = tokenizer(batch["input_text"], return_tensors="pt",
                                   padding=True, truncation=True, max_length=896).to("cuda")
            else:
                continue
            # 2. 前向传播
            labels = inputs.input_ids
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # 3. 梯度累积: 每次只反传 loss / accumulation_steps
            (loss / accumulation_steps).backward()

            # 4. 累计到一定步数，再执行一次优化
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            #5. 定时清理显存，防止OOM
            if global_step % 300 == 0:
                torch.cuda.empty_cache()
            # 5. 每 eval_interval 个 "优化步" 进行一次评估
            if global_step % 100 == 0:
                torch.cuda.empty_cache()

        #当前epoch结束后，评估dev_loss,并保存到csv文件中
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for dev_batch in dev_subset_dataloader:
                dev_inputs = tokenizer(dev_batch["input_text"],
                                        return_tensors="pt", padding=True, truncation=True, max_length=896).to("cuda")
                dev_labels = dev_inputs.input_ids
                dev_outputs = model(**dev_inputs, labels=dev_labels)
                total_loss += dev_outputs.loss.item()
        avg_loss = total_loss / len(dev_subset_dataloader)
        print(f"Epoch {epoch + 1},  Dev Loss: {avg_loss:.4f}")
        save_path = "/home/ubuntu/meditron-medmcqa-finetune/data/log/train_13.csv"
        log_dev_loss_to_csv(epoch+1, lora_rank, dropout, learning_rate, avg_loss, save_path)
        model.train()



    #评估准确率
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
    def evaluate_accuracy_batch(model, tokenizer, dev_loader):
        options = ["A", "B", "C", "D"]
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                prompts = batch["prompts"]  # list of str, 长度为 batch_size
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
        accuracy = correct / total if total > 0 else 0
        return accuracy

    model.eval()
    accuracy = evaluate_accuracy_batch(model, tokenizer, dev_loader)
    return accuracy


#记录每一轮结束时的dev_loss
def log_dev_loss_to_csv(epoch, lora_rank, dropout, lr, dev_loss, log_path):
    file_exists = Path(log_path).exists()
    with open(log_path, mode="a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "lora_rank", "dropout", "lr", "dev_loss", "accuracy"])  # 表头
        writer.writerow([epoch, lora_rank, dropout, lr, f"{dev_loss:.4f}", ""])
#记录训练结束后，在验证集上的准确率
def log_final_accuracy_to_csv(lora_rank, dropout, lr, accuracy, log_path):
    file_exists = Path(log_path).exists()
    with open(log_path, mode="a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "lora_rank", "dropout", "lr", "dev_loss", "accuracy"])
        writer.writerow(["final_accuracy", lora_rank, dropout, lr, "", f"{accuracy:.4f}"])

# ✅ Top 5 hyperparameter sets based on previous results
top_configs = [
    {"lora_rank": 16, "dropout": 0.1521, "lr": 0.000155},  # ✅ Trial 5
    {"lora_rank": 16, "dropout": 0.1840, "lr": 0.000080},  # ✅ Trial 13
    {"lora_rank": 16, "dropout": 0.1317, "lr": 0.000044},  # ✅ Trial 4
    {"lora_rank": 8,  "dropout": 0.2913, "lr": 0.000163},  # ✅ Trial 6
]



# ✅ Loop over top configs
for i, cfg in enumerate(top_configs):
    print(f"\n🚀 Running Trial {i} with lora_rank={cfg['lora_rank']}, dropout={cfg['dropout']}, lr={cfg['lr']:.6f}")
    score = train_model(
        lora_rank=cfg["lora_rank"],
        dropout=cfg["dropout"],
        learning_rate=cfg["lr"],
    )
    print(f"✅ Trial {i}: params={{'lora_rank': {cfg['lora_rank']}, 'dropout': {cfg['dropout']}, 'lr': {cfg['lr']:.6f}}}, score={score:.4f}")
    save_path = "/home/ubuntu/meditron-medmcqa-finetune/data/log/train_13.csv"
    log_final_accuracy_to_csv(
        lora_rank=cfg["lora_rank"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        accuracy=score,
        log_path=save_path
    )




