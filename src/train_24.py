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
import wandb



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 控制 cuDNN 层面
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(lora_rank=8, dropout=0.1, learning_rate=1e-4, alpha = 0.5, seed = 42):

    set_seed(seed)
    # ✅ 环境变量优化 CUDA 显存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # class DiscriminativeClassifier(nn.Module):
    #     def __init__(self, base_model: nn.Module, num_labels: int = 4):
    #         """
    #         base_model: 已加载好LoRA、8bit等配置的 Causal LM
    #         num_labels: 分类数目 (对MedMCQA为 4)
    #         """
    #         super().__init__()
    #         self.base_model = base_model
    #         self.num_labels = num_labels
    #
    #         # 从 base_model config 拿到 hidden_size
    #         self.hidden_size = base_model.config.hidden_size
    #
    #         # 线性分类头：输入 hidden_size，输出 num_labels
    #         self.classifier = nn.Linear(self.hidden_size, self.num_labels)
    #
    #     def mean_pooling(self, last_hidden_state, attention_mask):
    #         """简单的 mean pooling，根据 attention_mask 做加权平均。"""
    #         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    #         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    #         sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    #         return sum_embeddings / sum_mask
    #
    #     def forward(
    #             self,
    #             input_ids,
    #             attention_mask=None,
    #             labels=None,
    #             gold_label=None,
    #             kl_alpha=0.5
    #     ):
    #         """
    #         labels 期望是 soft label 分布, shape = (batch_size, 4).
    #         如果 labels=None, 就只返回 logits.
    #         """
    #         outputs = self.base_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             output_hidden_states=True,  # 关键：要让底层模型输出 hidden_states
    #             return_dict=True
    #         )
    #
    #         # 取最后一层 hidden state (B, seq_len, hidden_size)
    #         last_hidden_state = outputs.hidden_states[-1]
    #
    #         # 做 mean pooling (或自己改成取最后一个token / CLS位置等)
    #         pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
    #
    #         # 得到分类 logits (B, 4)
    #         logits = self.classifier(pooled_output)
    #
    #         # 训练模式：计算损失
    #         if labels is not None:
    #             # log-softmax 再和 labels 做 kl_div
    #             log_probs = F.log_softmax(logits, dim=-1)
    #             # KL 散度损失，要求第二个参数是目标分布（软标签）
    #             kl_loss = F.kl_div(log_probs, labels, reduction='batchmean')
    #             # 交叉熵损失，gold_label 是正确类别的整数表示
    #             ce_loss = F.cross_entropy(logits, gold_label)
    #             loss = kl_alpha * kl_loss + (1 - kl_alpha) * ce_loss
    #             return loss, logits
    #         else:
    #             # 推理模式：只返回 logits
    #             return logits

    class AttentionPooling(nn.Module):
        """
        单头 attention pooling:
        score = v^T tanh(W h_i)
        """

        def __init__(self, hidden_size: int, attn_hidden_size: int = 128, dropout: float = 0.1):
            super().__init__()
            self.W = nn.Linear(hidden_size, attn_hidden_size, bias=True)
            self.v = nn.Linear(attn_hidden_size, 1, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
            hidden_states = hidden_states.to(self.W.weight.dtype)
            # hidden_states: (B, L, H); attention_mask: (B, L)
            scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)  # (B, L)

            # 把 padding 位置设为 -inf，避免被选中
            scores = scores.masked_fill(attention_mask == 0, -1e4)

            attn_weights = F.softmax(scores, dim=-1)  # (B, L)
            attn_weights = self.dropout(attn_weights)  # 可选

            # (B, L, 1) * (B, L, H) → (B, H)
            pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
            return pooled  # (B, H)

    class DiscriminativeClassifier(nn.Module):
        def __init__(self, base_model: nn.Module, num_labels: int = 4,
                     attn_hidden_size: int = 128, attn_dropout: float = 0.15):
            super().__init__()
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size
            self.pooler = AttentionPooling(self.hidden_size, attn_hidden_size, attn_dropout)
            self.classifier = nn.Linear(self.hidden_size, num_labels)

        def forward(
                self,
                input_ids,
                attention_mask=None,
                labels=None,
                gold_label=None,
                kl_alpha=0.5
        ):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = outputs.hidden_states[-1]  # (B, L, H)
            pooled_output = self.pooler(last_hidden_state, attention_mask)
            logits = self.classifier(pooled_output)  # (B, 4)

            if labels is not None:
                log_probs = F.log_softmax(logits, dim=-1)
                kl_loss = F.kl_div(log_probs, labels, reduction='batchmean')
                ce_loss = F.cross_entropy(logits, gold_label)
                loss = kl_alpha * kl_loss + (1 - kl_alpha) * ce_loss
                return loss, logits
            else:
                return logits

    # ✅ 路径设置
    base_dir = Path("/home/ubuntu/meditron-medmcqa-finetune")
    model_path = base_dir / "models" / "meditron-7b"

    # 1) 加载 Tokenizer（本地）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # 2) 配置 BitsAndBytes (8-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    # 3) 加载原模型 (CausalLM) 到 8-bit
    base_causal_lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,  # 8-bit
        torch_dtype=torch.float16,
        local_files_only=True
    )
    # 4) 给原模型注入 LoRA adapter (这里一般是 Causal LM 的任务类型)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank*2,
        lora_dropout=dropout,
        task_type=TaskType.CAUSAL_LM
    )
    base_causal_lm = get_peft_model(base_causal_lm, lora_config)
    base_causal_lm.to("cuda")
    # 5) 构建判别式分类器 (带LoRA的 CausalLM + 分类头)
    model = DiscriminativeClassifier(base_model=base_causal_lm, num_labels=4)
    model.to("cuda")
    device = "cuda"

    # 6) 你是否要冻结除 LoRA 权重以外的参数
    for name, param in model.base_model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

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
    train_subset = train_dataset.select(range(30000))
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
        hard_labels = [ex["hard_label"] for ex in new_batch]  # list of int
        soft_labels = [ex["soft_label"] for ex in new_batch]  # list of list of float

        # 转成tensor
        hard_labels = torch.tensor(hard_labels, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels, dtype=torch.float32)

        return {
            "prompts": prompts,
            "hard_labels": hard_labels,
            "soft_labels": soft_labels
        }

    train_dataloader = DataLoader(train_subset, batch_size=3, shuffle=False, collate_fn=my_collate_fn)

    def evaluate_on_dev(
            model,
            tokenizer,
            dev_dataset,
            batch_size: int = 3,
            device: str = "cuda",
    ):
        """
        返回:
            acc      : float, 验证集整体准确率
            all_probs: Tensor (N,4), 每样本四选项概率
            all_preds: Tensor (N,) , 每样本预测类别索引 0‑3
            all_gold : Tensor (N,) , 每样本真实类别索引
        """

        # 把选项字母映射成 0‒3
        choice2idx = {"A": 0, "B": 1, "C": 2, "D": 3}

        def dev_collate_fn(batch):
            batch = [ex for ex in batch if ex is not None]
            if len(batch) == 0:
                raise ValueError("没有有效样本")
            # ⬇️ 1) prompts
            prompts = [ex["prompt"] for ex in batch]  # list[str]
            # ⬇️ 2) hard labels：把 "A"/"B"/… 转成 0‒3
            hard_labels = [choice2idx[ex["label"].strip()] for ex in batch]
            hard_labels = torch.tensor(hard_labels, dtype=torch.long)  # (B,)

            return {
                "prompts": prompts,  # list[str]
                "hard_labels": hard_labels,  # LongTensor (B,)
            }

        model.eval()
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dev_collate_fn,
        )

        all_probs, all_preds, all_gold = [], [], []
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in dev_loader:
                prompts = batch["prompts"]

                gold_labels = batch["hard_labels"].to(device)  # (B,)

                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=768,
                ).to(device)

                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )  # (B,4)

                probs = torch.softmax(logits, dim=-1)  # (B,4)
                preds = probs.argmax(dim=-1)  # (B,)

                # ✅ 直接在这里算 cross_entropy loss
                loss = F.cross_entropy(logits, gold_labels)
                batch_size = gold_labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_gold.append(gold_labels.cpu())

        # 拼接整个验证集
        all_probs = torch.cat(all_probs, dim=0)  # (N,4)
        all_preds = torch.cat(all_preds, dim=0)  # (N,)
        all_gold = torch.cat(all_gold, dim=0)  # (N,)
        # 计算准确率
        acc = (all_preds == all_gold).float().mean().item()
        # 计算整个 dev 集的平均 loss
        avg_dev_loss = total_loss / total_examples

        model.train()
        return acc, all_probs, all_preds, all_gold, avg_dev_loss

    # ✅ Optimizer
    from torch.optim import AdamW
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate
    )

    #引入 wanb，用来记录
    os.environ["WANDB_DIR"] = "/home/ubuntu/meditron-medmcqa-finetune/data"
    if wandb.run is not None:
        wandb.finish()
    wandb.init(
        project="medmcqa-attpooling-weightdecay-30k",
        name=f"lr{learning_rate:.6f}_dropout{dropout:.3f}_alpha_{alpha:.3f}_seed{seed}",
        config={
            "learning_rate": learning_rate,
            "dropout": dropout,
            "kl_alpha": alpha,
            "seed": seed,
        },
        settings=wandb.Settings(code_dir=".")  # 只跟踪代码，不自动同步大文件
    )
    # ✅ Training loop
    epochs = 4
    accumulation_steps = 5
    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            # --------- 1. 数据准备 ---------
            prompts = batch["prompts"]  # list[str]
            # 如果 options 已经在 prompt 里，就不用额外处理
            # tokenizer 自动批量编码并 Padding
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # soft_labels: float32, shape (B,4)
            soft_labels = batch["soft_labels"].float().to(device)
            # hard_labels: int64  , shape (B,)
            hard_labels = batch["hard_labels"].long().to(device)

            # --------- 2. 前向 + 反向 ---------
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=soft_labels,
                gold_label=hard_labels,  # 命名传参避免顺序错位
                kl_alpha=alpha
            )
            (loss/accumulation_steps).backward()
            total_loss += loss.item()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            wandb_save_step = 80
            if global_step % wandb_save_step == 0 and global_step != 0:
                avg_train_loss = total_loss / (wandb_save_step * accumulation_steps)
                wandb.log({"train_loss": avg_train_loss}, step=global_step)
                total_loss = 0.0

        if epoch >=0:
            save_path = base_dir / "data" / "log" / "train_24.csv"
            dev_acc, probs, preds, gold, dev_loss = evaluate_on_dev(
                model=model,
                tokenizer=tokenizer,
                dev_dataset=dev_eval_subset,
                batch_size=3,
                device="cuda",
            )
            print(f"Epoch: {epoch+1}, Dev accuracy: {dev_acc:.4f}")
            wandb.log({
                "dev_loss": dev_loss,
                "dev_acc": dev_acc
            }, step=global_step)
            log_final_accuracy_to_csv(epoch + 1, lora_rank, dropout, learning_rate, alpha, seed, dev_acc, save_path, 0)

    # save_path = base_dir / "data" / "log" / "train_24.csv"
    # dev_acc, probs, preds, gold, dev_loss = evaluate_on_dev(
    #     model=model,
    #     tokenizer=tokenizer,
    #     dev_dataset=dev_eval_subset,
    #     batch_size=3,
    #     device="cuda",
    # )
    # print(
    #     f"Final accuracy: {dev_acc:.4f} | "
    #     f"dropout={dropout:.3f}, "
    #     f"lr={learning_rate:.6f}, "
    #     f"alpha={alpha:.3f}, "
    #     f"seed={seed}"
    # )
    # log_final_accuracy_to_csv(epoch + 1, lora_rank, dropout, learning_rate, alpha, seed, dev_acc, save_path, 1)
    return dev_acc


def log_final_accuracy_to_csv(epoch, lora_rank, dropout, lr, alpha,seed, accuracy, log_path, is_final=0):
    file_exists = Path(log_path).exists()
    with open(log_path, mode="a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "lora_rank", "dropout", "lr", "alpha", "seed", "accuracy"])
        if not is_final:
            writer.writerow([epoch, lora_rank, dropout, lr, alpha, seed, f"{accuracy:.4f}"])
        else:
            writer.writerow(["final_accuracy", lora_rank, dropout, lr, alpha, seed, f"{accuracy:.4f}"])


top_configs = [
    {"lora_rank": 16, "dropout": 0.163, "lr": 7.7e-5,  "alpha": 0.44},

]
seed_list = [42]

# 3️⃣ 逐超参组合 × 逐 seed 训练 → 取均值
for i, cfg in enumerate(top_configs):
    print(f"\n🚀 Hyper‑Set {i} → lora_rank={cfg['lora_rank']}, "
          f"dropout={cfg['dropout']}, lr={cfg['lr']:.6f}, alpha={cfg['alpha']:.2f}")

    seed_scores = []      # 存放同一超参下，不同 seed 的验证准确率

    for sd in seed_list:
        print(f"    ▶ Seed {sd}...", end="", flush=True)

        score = train_model(                 # <‑‑ 你的训练函数
            lora_rank      = cfg["lora_rank"],
            dropout        = cfg["dropout"],
            learning_rate  = cfg["lr"],
            alpha          = cfg["alpha"],
            seed           = sd             # 关键：把 seed 传进去
        )

        seed_scores.append(score)
        print(f"  acc={score:.4f}")

    # 计算平均 / 方差
    mean_acc = float(np.mean(seed_scores))
    std_acc  = float(np.std(seed_scores))

    print(f"✅ Hyper‑Set {i}  mean‑acc={mean_acc:.4f}  std={std_acc:.4f}")


# from pathlib import Path
# import optuna
# import numpy as np
#
# # ✅ 设置日志保存目录
# log_dir = Path("/home/ubuntu/meditron-medmcqa-finetune/data/log")
# log_dir.mkdir(parents=True, exist_ok=True)
# db_path = log_dir / "train_24.db"
#
# # ✅ 固定3个种子
# seed_list = [42, 7, 123]
#
# # ✅ 目标函数：每组超参跑3个seed，取平均acc作为目标
# def objective(trial):
#     # 超参搜索空间
#     lr = trial.suggest_float("learning_rate", 7e-5, 1.2e-4, log=True)
#     alpha = trial.suggest_float("alpha", 0.25, 0.45)
#     dropout = trial.suggest_float("dropout", 0.12, 0.2)
#
#     acc_list = []
#
#     # 每个 seed 都独立训练一遍
#     for sd in seed_list:
#         score = train_model(
#             lora_rank=16,
#             dropout=dropout,
#             learning_rate=lr,
#             alpha=alpha,
#             seed=sd   # 传入不同seed
#         )
#         acc_list.append(score)
#
#     mean_score = float(np.mean(acc_list))
#
#     print(
#         f"Trial {trial.number}: "
#         f"params={{'lora_rank': {16}, 'dropout': {dropout:.3f}, 'lr': {lr:.6f}, 'alpha': {alpha:.3f}}}, "
#         f"mean_acc={mean_score:.4f}"
#     )
#
#     return mean_score   # 交给optuna的优化器去maximize
#
# # ✅ 使用 SQLite
# study = optuna.create_study(
#     direction="maximize",
#     study_name="meditron_lora_tuning",
#     storage=f"sqlite:///{db_path}",
#     load_if_exists=True
# )
#
# # ✅ 开始搜索
# try:
#     study.optimize(objective, n_trials=20, show_progress_bar=True)
# except KeyboardInterrupt:
#     print("🛑 手动中断调参，已保存当前进度。")
#
# # ✅ 最后输出结果
# print("🎯 最优参数:", study.best_params)
# print(f"✅ 最优平均准确率: {study.best_value:.4f}")