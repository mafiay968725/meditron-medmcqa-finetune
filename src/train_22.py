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
from accelerate import Accelerator
from torch.optim import AdamW
import optuna
import joblib
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def train_model(lora_rank=8, dropout=0.1, learning_rate=1e-4, alpha = 0.5):

    # ✅ 环境变量优化 CUDA 显存
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    # ✅ 路径设置
    base_dir = Path("/home/ubuntu/meditron-medmcqa-finetune")
    model_path = base_dir / "models" / "meditron-7b"

    # ✅ 加载 Tokenizer（本地）
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ✅ 加载 8-bit 模型（本地）
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)


    # ✅ 加载数据
    processed_data = load_from_disk(base_dir / "data" / "processed_dataset_soft_label")
    train_dataset = processed_data["train"]
    dev_dataset = processed_data["dev"]

    def format_example(example):
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        hard_label = label_map.get(example["label"], 0)

        option_prefix = ["A: ", "B: ", "C: ", "D: "]
        raw_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        options = [f"{prefix}{text}" for prefix, text in zip(option_prefix, raw_options)]


        return {
            "options": options,
            "hard_label": hard_label,
        }


    train_dataset = train_dataset.map(format_example)
    # 划分 train_eval_subset：从训练集划出 1000 条用于训练中评估准确率（early stopping）
    train_dataset = train_dataset.shuffle(seed=42)
    dev_dataset = dev_dataset.shuffle(seed=42)
    train_subset = train_dataset.select(range(10000))
    # 打乱验证集，划分出两个部分
    dev_eval_subset = dev_dataset.select(range(1000))  # ⬅️ 每轮评估准确率
    dev_final_subset = dev_dataset.select(range(1000, len(dev_dataset)))  # ⬅️ 最终评估准确率

    # ✅ collate_fn
    def my_collate_fn(batch):
        # batch: list[dict], 每个dict包含 prompt/options/hard_label/soft_label
        new_batch = []
        for ex in batch:
            if ex.get("second_choice") is None:
                ex["second_choice"] = ""
            if ex is not None:
                new_batch.append(ex)
        if len(new_batch) == 0:
            raise ValueError("没有有效样本")
        prompts = [ex["prompt"] for ex in new_batch]  # list of str
        options_list = [ex["options"] for ex in new_batch]  # list of list of str
        hard_labels = [ex["hard_label"] for ex in new_batch]  # list of int
        soft_labels = [ex["soft_label"] for ex in new_batch]  # list of list of float

        # 转成tensor
        hard_labels = torch.tensor(hard_labels, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels, dtype=torch.float32)

        return {
            "prompts": prompts,
            "options": options_list,
            "hard_labels": hard_labels,
            "soft_labels": soft_labels
        }


    device_map = {"": accelerator.local_process_index}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        local_files_only=True
    )

    lora_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=dropout, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)

    train_dataloader = DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=my_collate_fn)

    optimizer = AdamW(model.parameters(), lr=learning_rate)


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    device = accelerator.device


    # ✅ 准确率评估函数
    def evaluate_model_accuracy(model, tokenizer, dev_dataset, accelerator):
        def dev_collate_fn(batch):

            return {
                "prompts": [s.get("prompt", "") for s in batch],
                "gold_labels": [s.get("label", "") for s in batch],
                "opa": [s.get("opa", "") for s in batch],
                "opb": [s.get("opb", "") for s in batch],
                "opc": [s.get("opc", "") for s in batch],
                "opd": [s.get("opd", "") for s in batch],
            }

        dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)

        def compute_per_example_loss_after_answer(model, tokenizer, texts,max_length, device):
            # 1. 分词
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # ✅ 2. 直接复用训练时的 masking 函数！
            labels = mask_labels_before_answer(input_ids, tokenizer).to(device)

            # 4) 放到 GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

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

        torch.cuda.empty_cache()
        model.eval()
        device = accelerator.device

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dev_loader:
                prompts = batch["prompts"]
                labels = batch["gold_labels"]
                opa = batch["opa"]
                opb = batch["opb"]
                opc = batch["opc"]
                opd = batch["opd"]
                all_options = ["A", "B", "C", "D"]
                option_texts = [opa, opb, opc, opd]
                batch_size = len(prompts)
                if len(prompts) == 0:
                    continue  # 跳过空 batch

                candidate_texts = []
                for i in range(batch_size):
                    for j, opt in enumerate(all_options):
                        full_text = f"{prompts[i]} {opt}: {option_texts[j][i]}"
                        candidate_texts.append(full_text)

                losses = compute_per_example_loss_after_answer(
                    model, tokenizer, candidate_texts, max_length=768, device=device
                )
                losses = losses.view(batch_size, 4)

                preds = torch.argmin(losses, dim=1).tolist()
                all_preds.extend(preds)
                label_ids = [{"A": 0, "B": 1, "C": 2, "D": 3}[lbl] for lbl in labels]
                all_labels.extend(label_ids)

        all_preds = accelerator.gather_for_metrics(
            torch.tensor(all_preds, dtype=torch.long, device=accelerator.device)
        ).cpu()
        all_labels = accelerator.gather_for_metrics(
            torch.tensor(all_labels, dtype=torch.long, device=accelerator.device)
        ).cpu()

        correct = (all_preds == all_labels).sum().item()
        total = all_preds.size(0)
        accuracy = correct / total if total > 0 else 0
        return accuracy

    def _find_answer_pair_by_tokens(tokenizer, input_ids):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_seq = ["Answer", ":"]
        n, m = len(tokens), len(target_seq)
        for i in range(n - m + 1):
            if tokens[i:i + m] == target_seq:
                return i
        return None

    def mask_labels_before_answer(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
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
                print(f"[Warning] Sample {i} has no 'Answer:' token.")

        return masked_labels




    def compute_ce_kl_loss(
            model,
            input_ids: torch.Tensor,  # [batch_size*4, seq_len]
            attention_mask: torch.Tensor,  # [batch_size*4, seq_len]
            labels: torch.Tensor,  # [batch_size*4, seq_len]  (含 -100)
            hard_labels: torch.Tensor,  # [batch_size], 0~3
            soft_labels: torch.Tensor,  # [batch_size, 4], sum=1
            alpha: float = 0.5
    ):
        """
        将“Prompt + 解释 + 选项”共4条序列的 token-level loss 计算并汇总，
        得到各选项的平均 NLL -> 再做 (1−alpha)*CE + alpha*KL。

        参数：
        - model：自回归语言模型（GPT/LLaMA等）
        - input_ids / attention_mask：对 4*batch_size 条序列做了拼接
        - labels：mask掉不需要训练的 token（-100），只保留答案部分；形状同 input_ids
        - hard_labels：每条样本的正确选项下标 [batch_size]，如 0/1/2/3
        - soft_labels：teacher soft label分布 [batch_size,4]
        - alpha：KL权重

        返回：
        - loss：标量，用于 loss.backward()
        """

        # 1) 前向传播（不使用 model(..., labels=...)，避免默认reduction=mean）
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: [batch_size*4, seq_len, vocab_size]

        # 2) 对齐标签（shift），计算每个位置 token-level CE (reduction='none')
        #    注意只对 seq_len-1 位置做预测（自回归）
        shift_logits = logits[:, :-1, :].contiguous()  # [batch_size*4, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:].contiguous()  # [batch_size*4, seq_len-1]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_level_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # => shape [ (batch_size*4)*(seq_len-1) ]

        token_level_loss = token_level_loss.view(shift_labels.size())  # [batch_size*4, seq_len-1]

        # 3) 去除 -100 标记（不计入loss的prompt部分）
        valid_mask = (shift_labels != -100).float()
        sum_loss_per_seq = (token_level_loss * valid_mask).sum(dim=1)  # [batch_size*4]
        token_count = valid_mask.sum(dim=1)  # [batch_size*4]
        avg_nll_per_seq = sum_loss_per_seq / (token_count + 1e-8)  # [batch_size*4]

        # 4) reshape成 [batch_size, 4]
        batch_size = hard_labels.size(0)
        avg_nll_per_seq = avg_nll_per_seq.view(batch_size, 4)  # [batch_size,4]

        # 5) 把负对数似然取负值 => 作为 4个选项的"分数"
        #    score_for_each_option更大 => 说明选项更可能
        score_for_each_option = - avg_nll_per_seq  # [batch_size,4]

        # 6) 计算KL：KL(P_teacher || P_student)
        #    先对score_for_each_option做 log_softmax => log P_student
        log_student_probs = F.log_softmax(score_for_each_option, dim=-1)  # [batch_size,4]

        # teacher 软标签 [batch_size,4], sum=1
        kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
        kl_loss = kl_loss_fct(log_student_probs, soft_labels)

        # 7) 计算CE：把score_for_each_option当做4类分类logits
        ce_loss_fct = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fct(score_for_each_option, hard_labels)  # shape []

        # 8) 整合loss
        loss = (1 - alpha) * ce_loss + alpha * kl_loss

        return loss



    # ✅ Training loop
    epochs = 2
    global_step = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            prompts = batch["prompts"]  # list[str]
            options_list = batch["options"]  # list[list[str]]
            hard_labels = batch["hard_labels"].to(device)# [batch_size]
            soft_labels = batch["soft_labels"].to(device)  # [batch_size,4]

            # 把 batch_size 条数据每条4个选项 → 拼成 batch_size*4 个文本
            input_texts = []
            for idx, prompt_text in enumerate(prompts):
                for option_text in options_list[idx]:
                    # e.g. "Prompt + 解释 +  'Answer:' + 选项"
                    # 根据你实际逻辑拼起来即可
                    text = f"{prompt_text}{option_text}"
                    input_texts.append(text)

            # tokenizer
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=896
            )

            input_ids = inputs["input_ids"].to(device)  # [batch_size*4, seq_len]
            attention_mask = inputs["attention_mask"].to(device)

            # 利用你写好的函数，把 prompt/解释部分标签设成 -100，只保留 "Answer" 之类需要计算loss的地方
            labels = mask_labels_before_answer(input_ids, tokenizer).to(device)

            with accelerator.accumulate(model):
                loss = compute_ce_kl_loss(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    hard_labels=hard_labels,
                    soft_labels=soft_labels,
                    alpha=alpha  # or any
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        # ✅ epoch 结束时评估
        # model.eval()
        # save_path = base_dir / "data" / "log" / "train_22.csv"
        # if epoch >= 0:
        #     accuracy = evaluate_model_accuracy(model, tokenizer, dev_eval_subset, accelerator)
        #     print(f"Epoch {epoch + 1},  Accuracy: {accuracy:.4f}")
        #     log_final_accuracy_to_csv(epoch+1, lora_rank, dropout, learning_rate, accuracy, save_path,0)

    accuracy = evaluate_model_accuracy(model,tokenizer, dev_eval_subset, accelerator) #训练完成后，评估最终的准确率
    save_path = base_dir / "data" / "log" / "train_22.csv"
    log_final_accuracy_to_csv(epochs, lora_rank, dropout, learning_rate, accuracy, save_path, 1)

    del model
    del optimizer
    del train_dataloader
    torch.cuda.empty_cache()
    accelerator.free_memory()

    return accuracy


set_seed(42) #固定随机种子，
print("Using", torch.cuda.device_count(), "GPUs") #打印可用GPU数量

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


def main():
    # -------------------------------------------------------
    # 1) 初始化 Accelerator（多进程会在此 spawn）
    # -------------------------------------------------------
    accumulation_steps = 8
    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)

    # -------------------------------------------------------
    # 2) 设置数据库和日志路径
    # -------------------------------------------------------
    log_dir = Path("/home/ubuntu/meditron-medmcqa-finetune/data/log")
    log_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在就创建
    db_path = log_dir / "train_22.db"

    # -------------------------------------------------------
    # 3) 定义 Optuna 的 objective 函数
    #
    #    注意，这里的 objective 只在主进程真正发挥“调参、返回分数”
    #    非主进程也会进入，但我们会在里面做判断。
    # -------------------------------------------------------
    def objective(trial):
        # 仅主进程进行参数采样
        if accelerator.is_main_process:
            lr = trial.suggest_float("learning_rate", 2e-5, 1.2e-4, log=True)
            alpha = trial.suggest_float("alpha", 0.2, 0.8)
        else:
            # 非主进程占位（不会用于决定超参）
            lr = None
            alpha = None

        # 主进程广播超参给其它进程，以保证所有 GPU 一起用同样的超参训练
        [lr, alpha] = accelerator.broadcast_object_list([lr, alpha])

        # 调用你的训练函数
        score = train_model(
            lora_rank=16,
            dropout=0.15,
            learning_rate=lr,
            alpha=alpha
        )

        # 主进程打印信息；非主进程可以不打印
        if accelerator.is_main_process:
            print(f"Trial {trial.number}: params={{lr={lr:.6f}, alpha={alpha:.2f}}}, score={score:.4f}")

        return score

    # -------------------------------------------------------
    # 4) 只主进程创建（或加载） Study，以免重复写 DB
    # -------------------------------------------------------
    if accelerator.is_main_process:
        study = optuna.create_study(
            direction="maximize",
            study_name="meditron_lora_tuning",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)  # TPE 贝叶斯优化示例
        )
    accelerator.wait_for_everyone()

    # -------------------------------------------------------
    # 5) 所有进程都获取同一个 Study（共享同一个 storage）
    #    这样才能在多进程下正常同步
    # -------------------------------------------------------
    study = optuna.load_study(
        study_name="meditron_lora_tuning",
        storage=f"sqlite:///{db_path}"
    )

    # -------------------------------------------------------
    # 6) 仅主进程调用 study.optimize，真正做调参；其它进程只负责训练
    # -------------------------------------------------------
    if accelerator.is_main_process:
        study.optimize(objective, n_trials=10, show_progress_bar=True)

        # 打印最终结果
        print("🎯 最优参数:", study.best_params)
        print(f"✅ 最优准确率: {study.best_value:.4f}")

    accelerator.wait_for_everyone()

# -----------------------------------------------------------
# 入口点：确保多进程环境下只在 main 里做逻辑
# -----------------------------------------------------------
if __name__ == "__main__":
    main()

