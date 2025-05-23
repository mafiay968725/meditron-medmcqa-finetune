from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
from peft import PeftModel

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

from datasets import load_from_disk

processed_data = load_from_disk("/home/ubuntu/meditron-medmcqa-finetune/data/processed_dataset")

# 里面包含 train/dev/test 分割
train_dataset = processed_data["train"]
dev_dataset = processed_data["dev"]

def format_example(example):
    # 如果你之前在 "prompt" 字段已经包含了 "Answer: ???"
    # 并且 "label" 是 "A/B/C/D"
    # 这里直接把它拼到 prompt 后面即可
    text = example["prompt"] + " " + example["label"]  # 例如: "...Answer: C"
    return {"input_text": text}





# 3) 设置Lora配置
lora_config = LoraConfig(
    r=8,  # Rank of the adapter
    lora_alpha=16,  # Lora scaling factor
    lora_dropout=0.1,  # Dropout
    task_type=TaskType.CAUSAL_LM,  # Causal language modeling
)

# 4) 构建Lora模型
model = get_peft_model(model, lora_config)
model.to("cuda")  # 确保模型在GPU上
model.print_trainable_parameters()  # 看可训练参数数量

# 5) 数据加载器
train_dataset = train_dataset.map(format_example)
dev_dataset = dev_dataset.map(format_example)
train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
train_subset = train_dataset #使用全集
dev_subset = dev_dataset.shuffle(seed=42).select(range(1000)) #先用验证集的一部分进行计算

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

    return torch.utils.data.dataloader.default_collate(filtered_batch)
# 然后在 DataLoader 中使用：
train_dataloader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
dev_dataloader = DataLoader(dev_subset, batch_size=4, collate_fn=my_collate_fn)


# 6) 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=1e-4)


# 7) 训练循环
eval_interval = 1250  # 每1250次优化后评估一次
epochs = 2
best_dev_loss = float("inf")  # 用来保存当前最小的验证集损失

accumulation_steps = 4  # 每4个mini-batch累积一次梯度 => 有效batch_size=8×2=16
global_step = 0         # 记录真实的优化步数（每完成一次optimizer.step()就+1）

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # 确保每个epoch开始时梯度为0

    for i, batch in enumerate(train_dataloader):
        # 1. 准备输入
        if "input_text" in batch:
            inputs = tokenizer(batch["input_text"], return_tensors="pt",
                               padding=True, truncation=True, max_length = 896).to("cuda")
        else:
            print("❌ 缺失 input_text 的样本：", batch)
            sys.exit("⛔ 程序已终止，因为有样本缺失 input_text")

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

            # 5. 每 eval_interval 个 "优化步" 进行一次评估
            if global_step % 100 == 0:
                torch.cuda.empty_cache()
            if global_step % eval_interval == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for dev_batch in dev_dataloader:
                        dev_inputs = tokenizer(dev_batch["input_text"],
                                               return_tensors="pt", padding=True, truncation=True, max_length = 896).to("cuda")
                        dev_labels = dev_inputs.input_ids
                        dev_outputs = model(**dev_inputs, labels=dev_labels)
                        total_loss += dev_outputs.loss.item()
                avg_loss = total_loss / len(dev_dataloader)
                print(f"Epoch {epoch + 1}, Global Step {global_step}, Dev Loss: {avg_loss:.4f}")

                # 保存最优模型
                if avg_loss < best_dev_loss:
                    best_dev_loss = avg_loss
                    model.save_pretrained("/home/ubuntu/meditron-medmcqa-finetune/data/train_9/best", safe_serialization=True)
                    print(f"💾 最优模型已保存，当前 Dev Loss: {avg_loss:.4f}")
                model.train()

    # 每个 epoch 结束后保存一次模型
    save_path = base_dir / "data" / "train_9" / f"epoch_{epoch + 1}"
    model.save_pretrained(save_path, safe_serialization=True)
    if epoch == 0:
        tokenizer.save_pretrained("/home/ubuntu/meditron-medmcqa-finetune/data/train_9/tokenizer")
    print(f"✅ 模型已保存至 {save_path}")