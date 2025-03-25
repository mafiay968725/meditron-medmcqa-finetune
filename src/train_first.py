import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import sys


model_name_or_path =  "/root/meditron-medmcqa-finetune/models/meditron-7b" # 或者你本地路径

# 1) 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
# GPT/LLAMA 系模型通常用左侧 padding
tokenizer.pad_token = tokenizer.eos_token  # 避免出现警告

# 2) 配置 8-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,          # 8bit 量化
    llm_int8_threshold=6.0,     # 一些默认阈值
    llm_int8_has_fp16_weight=False,
)

# 3) 加载模型 (8-bit)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto"  # 根据显存自动分配到 GPU
)


from datasets import load_from_disk

processed_data = load_from_disk("/root/meditron-medmcqa-finetune/data/processed_dataset")
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

# 5) 数据加载器
train_dataset = train_dataset.map(format_example)
dev_dataset = dev_dataset.map(format_example)
train_dataset = train_dataset.filter(lambda x: x is not None and "input_text" in x)
dev_dataset = dev_dataset.filter(lambda x: x is not None and "input_text" in x)
train_subset = train_dataset.select(range(10000)) #构建一个10k的子训练集，进行试验

from torch.utils.data._utils.collate import default_collate
def my_collate_fn(batch):
    for sample in batch:
        if sample is not None:
            # 将 None 替换为默认值，例如空字符串
            if sample.get("topic_name") is None:
                sample["topic_name"] = ""
            if sample.get("exp") is None:
                sample["exp"] = ""
    # 打印调试信息：检查每个样本是否存在 None 或缺失必须的键
    for i, sample in enumerate(batch):
        if sample is None:
            print(f"样本 {i} 是 None")
        else:
            for key, value in sample.items():
                if value is None:
                    print(f"样本 {i} 中键 {key} 的值为 None")
    # 过滤掉整体为 None 的样本
    filtered_batch = [sample for sample in batch if sample is not None]
    if len(filtered_batch) == 0:
        raise ValueError("过滤后，当前批次没有有效样本，请检查数据预处理逻辑")

    return torch.utils.data.dataloader.default_collate(filtered_batch)
# 然后在 DataLoader 中使用：
train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=my_collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=my_collate_fn)


# 6) 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)


# 7) 训练循环
eval_interval = 200  # 每200个batch评估一次
epochs = 1
for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        if "input_text" in batch:
            inputs = tokenizer(batch["input_text"], return_tensors="pt", padding=True, truncation=True).to("cuda")
        else:
            print("❌ 缺失 input_text 的样本：", batch)
            sys.exit("⛔ 程序已终止，因为有样本缺失 input_text")
        labels = inputs.input_ids
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 每eval_interval个batch进行一次评估
        if (i + 1) % eval_interval == 0:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for dev_batch in dev_dataloader:
                    dev_inputs = tokenizer(dev_batch["input_text"], return_tensors="pt", padding=True, truncation=True).to("cuda")
                    dev_labels = dev_inputs.input_ids
                    dev_outputs = model(**dev_inputs, labels=dev_labels)
                    total_loss += dev_outputs.loss.item()
            avg_loss = total_loss / len(dev_dataloader)
            print(f"Epoch {epoch + 1}, Step {i + 1}, Dev Loss: {avg_loss}")
            model.train()
