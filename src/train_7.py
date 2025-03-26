import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
from torch.cuda.amp import autocast, GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 启用 PyTorch 的更智能显存分配策略
model_name_or_path = "/root/meditron-medmcqa-finetune/models/meditron-7b"  # 或者你本地路径

# 1) 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # 避免出现警告

# 2) 加载模型，使用 FP16 混合精度（替换原来的 8-bit 量化）
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # 使用 FP16 加载，降低显存占用
    device_map="auto"  # 根据显存自动分配到 GPU
)

from datasets import load_from_disk

processed_data = load_from_disk("/root/meditron-medmcqa-finetune/data/processed_dataset")
# 里面包含 train/dev/test 分割
train_dataset = processed_data["train"]
dev_dataset = processed_data["dev"]


def format_example(example):
    """
    假设 example 中包含以下字段：
    - "prompt": 题干和选项 (字符串)
    - "label":  "A"/"B"/"C"/"D"
    - "exp":    可能为空，或者是专家思路 (字符串)

    将它们组合成一段 'input_text'，包含：
      - 原始 prompt
      - CoT 提示: "Let's think step by step."
      - 如果 exp 不为空，则加入专家思路
      - 最后附加 "Answer: X"
    """
    text = example["prompt"]  # 例: "Question: ... \nOptions: ... \nAnswer:"
    label = example["label"]  # 例: "C"

    # 拼接 CoT 提示
    text += "\nLet's think step by step."

    # 如果 exp 不为空（并且不只是空格），则加上专家思路
    if (example.get("exp") is not None) and example["exp"].strip():
        text += f"\nExpert explanation: {example['exp']}"

    # 最后再补上“Answer: X”
    text += f"\nAnswer: {label}"

    return {"input_text": text}


# 3) 设置 LoRA 配置
lora_config = LoraConfig(
    r=8,  # adapter 的 Rank
    lora_alpha=16,  # 缩放因子
    lora_dropout=0.1,  # Dropout 概率
    task_type=TaskType.CAUSAL_LM,  # 因果语言建模任务
)

# 4) 构建 LoRA 模型
model = get_peft_model(model, lora_config)
model.to("cuda")  # 确保模型在 GPU 上
model.print_trainable_parameters()  # 打印可训练参数数量

# 5) 数据加载器
train_dataset = train_dataset.map(format_example)
dev_dataset = dev_dataset.map(format_example)
train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
train_subset = train_dataset.select(range(10000))  # 构建一个 10k 的子训练集
dev_subset = dev_dataset.shuffle(seed=42).select(range(1000))  # 先用验证集的一部分进行计算

from torch.utils.data._utils.collate import default_collate


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
    return default_collate(filtered_batch)


train_dataloader = DataLoader(train_subset, batch_size=6, shuffle=True, collate_fn=my_collate_fn)
dev_dataloader = DataLoader(dev_subset, batch_size=6, collate_fn=my_collate_fn)

# 6) 优化器
optimizer = AdamW(model.parameters(), lr=1e-4)

# 7) 训练循环设置
eval_interval = 300  # 每 300 次优化步评估一次
epochs = 3
best_dev_loss = float("inf")  # 保存当前最小的验证集损失
accumulation_steps = 3  # 每 2 个 mini-batch 累积一次梯度，实际 batch_size 为 8×2=16
global_step = 0  # 记录真实的优化步数

# 初始化 AMP 的 GradScaler
scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # 每个 epoch 开始时梯度清零

    for i, batch in enumerate(train_dataloader):
        # 1. 准备输入
        if "input_text" in batch:
            inputs = tokenizer(batch["input_text"], return_tensors="pt",
                               padding=True, truncation=True, max_length=1024).to("cuda")
        else:
            print("❌ 缺失 input_text 的样本：", batch)
            sys.exit("⛔ 程序已终止，因为有样本缺失 input_text")

        # 2. 前向传播，使用 AMP 自动混合精度
        with autocast():
            labels = inputs.input_ids
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

        # 3. 梯度累积：反传 loss/accumulation_steps
        scaler.scale(loss / accumulation_steps).backward()

        # 4. 当达到累积步数后，进行一次优化更新
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            # 5. 每 eval_interval 次优化步评估一次模型
            if global_step % eval_interval == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for dev_batch in dev_dataloader:
                        dev_inputs = tokenizer(dev_batch["input_text"],
                                               return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
                            "cuda")
                        dev_labels = dev_inputs.input_ids
                        with autocast():
                            dev_outputs = model(**dev_inputs, labels=dev_labels)
                        total_loss += dev_outputs.loss.item()
                avg_loss = total_loss / len(dev_dataloader)
                print(f"Epoch {epoch + 1}, Global Step {global_step}, Dev Loss: {avg_loss:.4f}")

                if avg_loss < best_dev_loss:
                    best_dev_loss = avg_loss
                    model.save_pretrained("/root/meditron-medmcqa-finetune/data/train_7/best")
                    print(f"💾 最优模型已保存，当前 Dev Loss: {avg_loss:.4f}")
                torch.cuda.empty_cache()
                model.train()

    # 每个 epoch 结束后保存一次模型
    save_path = f"/root/meditron-medmcqa-finetune/data/train_7/epoch_{epoch + 1}"
    model.save_pretrained(save_path)
    if epoch == 0:
        tokenizer.save_pretrained("/root/meditron-medmcqa-finetune/data/train_7/tokenizer")
    print(f"✅ 模型已保存至 {save_path}")
