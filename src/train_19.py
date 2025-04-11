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
        label_to_opt = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
        # 拼接 prompt, label, 冒号和对应选项内容
        text = example["prompt"] + " " + example["label"] + ": " + example[label_to_opt[example["label"]]]
        return {"input_text": text}

    train_dataset = train_dataset.map(format_example)
    dev_dataset = dev_dataset.map(format_example)
    train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
    dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
    # 划分 train_eval_subset：从训练集划出 1000 条用于训练中评估准确率（early stopping）
    train_dataset = train_dataset.shuffle(seed=42)
    dev_dataset = dev_dataset.shuffle(seed=42)
    train_subset = train_dataset.select(range(10000))
    # 打乱验证集，划分出两个部分
    dev_eval_subset = dev_dataset.select(range(1000))  # ⬅️ 每轮评估准确率
    dev_final_subset = dev_dataset.select(range(1000, len(dev_dataset)))  # ⬅️ 最终评估准确率



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
    dev_eval_dataloader = DataLoader(dev_eval_subset, batch_size=3, shuffle=True, collate_fn=my_collate_fn)

    # ✅ 准确率评估函数
    def evaluate_model_accuracy(model, tokenizer, dev_dataset):
        def dev_collate_fn(batch):
            for sample in batch:
                sample["topic_name"] = sample.get("topic_name", "")
                sample["exp"] = sample.get("exp", "")

            return {
                "prompts": [s.get("prompt", "") for s in batch],
                "gold_labels": [s.get("label", "") for s in batch],
                "opa": [s.get("opa", "") for s in batch],
                "opb": [s.get("opb", "") for s in batch],
                "opc": [s.get("opc", "") for s in batch],
                "opd": [s.get("opd", "") for s in batch],
            }

        dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)

        def compute_per_example_loss_after_answer(model, tokenizer, texts,max_length=768):
            """
            在验证/推断阶段，对输入的多条文本，只计算从“Answer:”开始的token的平均loss，
            其它部分（问句、选项列表等）设为 -100 不纳入CE损失。

            参数:
              model: 你的LoRA微调后的Causal LM模型
              tokenizer: 对应的分词器 (不会变动)
              texts: list[str]，长度 = batch_size，也可能是4倍batch_size (每个样本4选项)
              answer_token_ids: tokenizer.encode("Answer:", add_special_tokens=False)
              max_length: 分词长度上限

            返回:
              per_example_loss: Tensor，形状 [batch_size]，表示每条文本的平均loss
            """
            # 1. 分词
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # ✅ 2. 直接复用训练时的 masking 函数！
            labels = mask_labels_before_answer(input_ids, tokenizer)

            # 4) 放到 GPU
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            # 5) 前向传播 (不使用 outputs.loss，手动拿 logits 计算更灵活)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits  # shape: [B, L, vocab_size]

            # 6) 做 shift：Causal LM 通常要对 logits[:-1] 和 labels[1:]对齐
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()  # shape: [B, L-1]

            # 7) 自定义 token-level cross entropy
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss_tokens = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # 把它 reshape 回 [B, L-1]
            loss_tokens = loss_tokens.view(shift_labels.size())

            # 8) 只对 label != -100 的位置求和，再除以有效token数
            valid_mask = (shift_labels != -100).float()
            per_example_loss = (loss_tokens * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

            return per_example_loss


        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                prompts = batch["prompts"]
                labels = batch["gold_labels"]
                opa = batch["opa"]
                opb = batch["opb"]
                opc = batch["opc"]
                opd = batch["opd"]
                all_options = ["A", "B", "C", "D"]
                option_texts = [opa, opb, opc, opd]  # 每个都是 list[str]，长度为 batch_size
                batch_size = len(prompts)
                candidate_texts = [] # 构造 candidate_texts：每个样本 4 个句子，总共 batch_size × 4 个句子

                for i in range(batch_size):
                    for j, opt in enumerate(all_options):
                        option_content = option_texts[j][i]  # opa[i], opb[i], ...
                        full_text = f"{prompts[i]} {opt}: {option_content}"
                        candidate_texts.append(full_text)
                losses = compute_per_example_loss_after_answer(
                    model,
                    tokenizer,
                    candidate_texts,  # e.g.  batch_size * 4 个句子
                    max_length=768
                )
                losses = losses.view(batch_size, 4) # reshape 回 [batch_size, 4]
                preds = torch.argmin(losses, dim=1).cpu().numpy()
                pred_labels = [all_options[i] for i in preds]
                correct += sum(p == g for p, g in zip(pred_labels, labels))
                total += len(labels)
        return correct / total if total else 0

    def mask_labels_before_answer(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        """
        对 batch 内的每个样本，在 input_ids 中找到 `Answer:` 的起始位置，
        将该位置之前（含“Answer:”本身）所有 tokens 的 label 设为 -100，以便只对答案主体部分计算loss。

        参数:
          input_ids: [batch_size, seq_len] 的张量
          tokenizer: 你的分词器对象
          answer_tokens: 形如 tokenizer.encode("Answer:", add_special_tokens=False)

        返回:
          masked_labels: [batch_size, seq_len]，只有 “Answer:” 之后部分为原 token id，其余设为 -100
        """
        batch_size, seq_len = input_ids.size()
        # 先复制一份 input_ids 作为 labels
        masked_labels = input_ids.clone()

        # 遍历 batch 中每个样本
        for i in range(batch_size):
            row_ids = input_ids[i].tolist()
            start_idx = _find_answer_pair_by_tokens(tokenizer, row_ids)

            if start_idx is not None:
                end_of_answer_prefix = start_idx + 2  # 因为是 Answer + : 两个 token
                masked_labels[i, :end_of_answer_prefix] = -100 #Answer: 前的全部mask
            else:
                pass
            if start_idx is None:
                # print("Answer token ids:", answer_token_ids)
                # print("Example row (raw tokens):", tokenizer.convert_ids_to_tokens(row_ids))
                #
                # # ✅ 新增：打印 token 的 ascii 表示，方便发现不可见字符
                # print("Example row (ascii tokens):")
                # tokens = tokenizer.convert_ids_to_tokens(row_ids)
                # for idx, tk in enumerate(tokens):
                #     print(f"  {idx:03d}: {ascii(tk)}")  # 补齐位数更方便看
                print(f"[Warning] Sample {i} has no 'Answer:' token.")

        return masked_labels


    def _find_answer_pair_by_tokens(tokenizer, input_ids):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_seq = ["Answer", ":"]
        n, m = len(tokens), len(target_seq)
        for i in range(n - m + 1):
            if tokens[i:i + m] == target_seq:
                return i
        return None

    # ✅ Optimizer
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # ✅ Training loop
    epochs = 2
    accumulation_steps = 5
    global_step = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            inputs = tokenizer(batch["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=896).to("cuda")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            # 根据需要将 "Answer:" 之前的部分mask掉
            labels = mask_labels_before_answer(input_ids, tokenizer).to(model.device)
            # 将注意力掩码也要带上
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            (outputs.loss / accumulation_steps).backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step % 300 == 0:
                torch.cuda.empty_cache()

        # # ✅ epoch 结束时评估
        # model.eval()
        # save_path = base_dir / "data" / "log" / "train_19.csv"
        # if epoch >= 0:
        #     accuracy = evaluate_model_accuracy(model, tokenizer, dev_eval_subset)
        #     print(f"Epoch {epoch + 1},  Accuracy: {accuracy:.4f}")
        #     log_final_accuracy_to_csv(epoch+1, lora_rank, dropout, learning_rate, accuracy, save_path,0)

    accuracy = evaluate_model_accuracy(model,tokenizer, dev_eval_subset) #训练完成后，评估最终的准确率
    save_path = base_dir / "data" / "log" / "train_19.csv"
    log_final_accuracy_to_csv(epochs, lora_rank, dropout, learning_rate, accuracy, save_path, 1)
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
def log_final_accuracy_to_csv(epoch, lora_rank, dropout, lr, accuracy, log_path, is_final=0):
    file_exists = Path(log_path).exists()
    with open(log_path, mode="a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "lora_rank", "dropout", "lr", "dev_loss", "accuracy"])
        if not is_final:
            writer.writerow([epoch, lora_rank, dropout, lr, "", f"{accuracy:.4f}"])
        else:
            writer.writerow(["final_accuracy", lora_rank, dropout, lr, "", f"{accuracy:.4f}"])

import optuna
import joblib
from pathlib import Path

# ✅ 设置保存路径
log_dir = Path("/home/ubuntu/meditron-medmcqa-finetune/data/log")
log_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在就创建

# ✅ 设定数据库和文件名
db_path = log_dir / "train_19.db"


def objective(trial):
    lr = trial.suggest_float("learning_rate", 5e-5, 1.7e-4, log=True)
    dropout = trial.suggest_float("dropout",0.1,0.25)
    score = train_model(
        lora_rank=16,
        dropout=dropout,
        learning_rate=lr,
    )

    print(
        f"Trial {trial.number}: params={{'lora_rank': {16}, 'dropout': {dropout}, 'lr': {lr:.6f}}}, score={score:.4f}")
    return score

# ✅ 使用 SQLite 存储，保存至指定路径
study = optuna.create_study(
    direction="maximize",
    study_name="meditron_lora_tuning",
    storage=f"sqlite:///{db_path}",
    load_if_exists=True
)

try:
    study.optimize(objective, n_trials=20, show_progress_bar=True)
except KeyboardInterrupt:
    print("🛑 手动中断调参，已保存当前进度。")

# ✅ 输出并保存
print("🎯 最优参数:", study.best_params)
print(f"✅ 最优准确率: {study.best_value:.4f}")

