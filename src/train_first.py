import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

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

train_dataset = train_dataset.map(format_example)
dev_dataset = dev_dataset.map(format_example)
train_dataset = train_dataset.filter(lambda x: "input_text" in x)
dev_dataset = dev_dataset.filter(lambda x: "input_text" in x)
train_subset = train_dataset.select(range(10000)) #构建一个10k的子训练集，进行试验

# 一个简单的 DataCollator，把 input_text -> tokenized
from transformers import DefaultDataCollator
import sys
def simple_data_collator(batch):
    texts = []
    for x in batch:
        if "input_text" in x:
            texts.append(x["input_text"])
        else:
            print("❌ 缺失 input_text 的样本：", x)
            for i, example in enumerate(train_subset):
                if "input_text" not in example:
                    print(f"Sample {i} is missing input_text: {example}")
                    break
            else:
                print("all samples have input_text")
            sys.exit("⛔ 程序已终止，因为有样本缺失 input_text")

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 可根据显存调大/调小
    )
    # 语言模型的 label 与 input_ids 相同
    tokenized["labels"] = tokenized["input_ids"].detach().clone()
    return tokenized

data_collator = simple_data_collator

from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# 1) LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # 因为这是一个 Causal Language Model
)

# 2) 将原模型转换为 PEFT (LoRA) 模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 看可训练参数数量

# 3) 定义训练参数
# training_args = TrainingArguments(
#     output_dir="./lora-meditron7b-checkpoints",
#     overwrite_output_dir=True,
#     num_train_epochs=1,               # 先试跑1轮
#     per_device_train_batch_size=1,    # 视显存调整
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=16,   # 累积梯度减少显存
#     evaluation_strategy="steps",      # 或者 "epoch"
#     eval_steps=500,                   # 多少 step 评估一次
#     save_steps=500,                   # 多少 step 保存一次
#     logging_steps=50,
#     learning_rate=1e-4,               # LoRA下常用 1e-4 ~ 2e-4
#     fp16=True,                        # 混合精度
#     optim="adamw_torch",
#     report_to="none"                  # 不用wandb可这么写
# )
training_args = TrainingArguments(
    output_dir="./models/checkpoints_first",
    overwrite_output_dir=True,
    num_train_epochs=1,               # 试跑1轮
    per_device_train_batch_size=1,    # 视显存调整
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # 累积梯度减少显存
    evaluation_strategy="epoch",      # 每个epoch结束时评估
    save_strategy="epoch",            # 每个epoch结束时保存检查点
    logging_steps=50,
    learning_rate=1e-4,               # LoRA下常用 1e-4 ~ 2e-4
    fp16=True,                        # 混合精度
    optim="adamw_torch",
    report_to="none"
)


# 4) 构造 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,#先使用10k条数据试验
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

# 5) 开始训练
trainer.train()