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
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)



def train_model(lora_rank=8, dropout=0.1, learning_rate=1e-4):

    # ✅ 环境变量优化 CUDA 显存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ✅ 路径设置
    base_dir = Path("/home/ubuntu/meditron-medmcqa-finetune")
    model_path = base_dir / "models" / "meditron-7b"

    # ✅ 加载 Tokenizer（本地）
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ✅ 加载 8-bit 模型（本地）
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,
                                                 device_map={"": 0}, local_files_only=True)

    # ✅ 加载数据
    processed_data = load_from_disk(base_dir / "data" / "processed_dataset_deepseek")
    train_dataset = processed_data["train"]
    dev_dataset = processed_data["dev"]

    def format_example(example):
        return {"input_text": example["prompt"] + " " + example["label"]}

    train_dataset = train_dataset.map(format_example)
    dev_dataset = dev_dataset.map(format_example)
    train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
    dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
    train_subset = train_dataset.shuffle(seed=42).select(range(10000))
    dev_subset = dev_dataset.shuffle(seed=42).select(range(1000))

    # ✅ 构建 Lora 模型
    lora_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=dropout, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config).to("cuda")

    # ✅ collate_fn
    def my_collate_fn(batch):
        for sample in batch:
            if sample.get("topic_name") is None:
                sample["topic_name"] = ""
            if sample.get("exp") is None:
                sample["exp"] = ""
        batch = [s for s in batch if s is not None]
        if not batch:
            raise ValueError("过滤后没有有效样本")
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataloader = DataLoader(train_subset, batch_size=3, shuffle=True, collate_fn=my_collate_fn)
    dev_subset_dataloader = DataLoader(train_subset, batch_size=3, shuffle=True, collate_fn=my_collate_fn)

    # ✅ 准确率评估函数（提前放置）
    def evaluate_model_accuracy(model, tokenizer, dev_dataset):
        def dev_collate_fn(batch):
            for sample in batch:
                sample["topic_name"] = sample.get("topic_name", "")
                sample["exp"] = sample.get("exp", "")
            return {"prompts": [s["prompt"] for s in batch], "gold_labels": [s["label"] for s in batch]}

        dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)

        def compute_per_example_loss(model, tokenizer, texts):
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits, labels = outputs.logits, inputs["input_ids"]
            shift_logits, shift_labels = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss_tokens = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view_as(shift_labels)
            mask = (shift_labels != tokenizer.pad_token_id).float()
            return (loss_tokens * mask).sum(1) / (mask.sum(1) + 1e-8)

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                prompts, labels = batch["prompts"], batch["gold_labels"]
                all_options = ["A", "B", "C", "D"]
                candidate_texts = [f"{p} {o}" for p in prompts for o in all_options]
                losses = compute_per_example_loss(model, tokenizer, candidate_texts).view(len(prompts), 4)
                preds = torch.argmin(losses, dim=1).cpu().numpy()
                pred_labels = [all_options[i] for i in preds]
                correct += sum(p == g for p, g in zip(pred_labels, labels))
                total += len(labels)
        return correct / total if total else 0

    # ✅ Optimizer
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # ✅ Training loop
    epochs = 4
    accumulation_steps = 5
    global_step = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            inputs = tokenizer(batch["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=896).to("cuda")
            outputs = model(**inputs, labels=inputs.input_ids)
            (outputs.loss / accumulation_steps).backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step % 300 == 0:
                torch.cuda.empty_cache()

        # ✅ epoch 结束时评估
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for dev_batch in dev_subset_dataloader:
                dev_inputs = tokenizer(dev_batch["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=896).to("cuda")
                dev_outputs = model(**dev_inputs, labels=dev_inputs.input_ids)
                total_loss += dev_outputs.loss.item()
        avg_loss = total_loss / len(dev_subset_dataloader)

        save_path = base_dir / "data" / "log" / "train_13.csv"
        log_dev_loss_to_csv(epoch + 1, lora_rank, dropout, learning_rate, avg_loss, save_path)
        if epoch > 0:
            accuracy = evaluate_model_accuracy(model, tokenizer, dev_subset)
            print(f"Epoch {epoch + 1}, Dev Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            log_final_accuracy_to_csv(lora_rank, dropout, learning_rate, accuracy, save_path)

    return accuracy


set_seed(42) #固定随机种子，

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
        writer.writerow(["accuracy", lora_rank, dropout, lr, "", f"{accuracy:.4f}"])

# ✅ Top 5 hyperparameter sets based on previous results
top_configs = [
    {"lora_rank": 16, "dropout": 0.1521, "lr": 0.000155},  # ✅ Trial 5
    {"lora_rank": 16, "dropout": 0.1840, "lr": 0.000080},  # ✅ Trial 13
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




