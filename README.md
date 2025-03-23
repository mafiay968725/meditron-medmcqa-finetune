# 🧠 Meditron-7B Fine-tuning for Medical Reasoning (MedMCQA)
Fine-tuning Meditron-7B for Multiple-Choice Medical QA Reasoning

本项目旨在微调强大的开源医学大语言模型 **Meditron-7B**，以提升其在 **医学推理题（MedMCQA）** 上的预测准确率。通过精细设计的 prompt 格式与参数高效微调方法（PEFT，如 LoRA、Adapter Tuning），探索不同微调策略对模型推理能力的提升效果。

---

## 📌 项目背景与目标

- **模型背景**：
  - [Meditron-7B](https://huggingface.co/epfl-llm/meditron-7b) 是基于 LLaMA-2-7B，通过医学语料（PubMed、医学指南等）继续预训练的开源大语言模型，适用于医疗问答、推理、临床文本分析等任务。
  
- **任务数据集**：
  - [MedMCQA](https://github.com/medmcqa/medmcqa) 是一项大规模医学多项选择问答数据集，涵盖 NEET PG 医学考试相关内容，包括：生理、药理、病理、内科、外科、儿科等。

- **项目目标**：
  - 微调 Meditron-7B，使其在 MedMCQA 数据集上获得更高的准确率，超越原始模型和其他基线方法。

---

## 🧹 数据预处理流程

1. 构建 Prompt 格式：

2. 构造标签：将 `cop` 字段映射为 A/B/C/D

3. 使用 `datasets.Dataset.from_dict()` 构建 HuggingFace `DatasetDict`

4. 样本清洗：去除空问题、无效样本、重复数据

---

## 🧠 模型加载与配置

| 步骤 | 工具 | 描述 |
|------|------|------|
| 模型加载 | `transformers.AutoModelForCausalLM` | 加载 Meditron-7B |
| 分词器加载 | `AutoTokenizer` | 注意 padding、特殊符号一致性 |
| 微调方法 | `LoRA`（通过 PEFT） | 显存高效，训练稳定 |
| 8-bit 量化 | `bitsandbytes` | 减少显存占用（可选） |
| 参数配置 | `Trainer` 或 `SFTTrainer` | 控制学习率、batch size、epoch、评估策略等 |

---

## 🏋️‍♀️ 模型训练与微调

- 构建训练子集（如每轮 10k 样本）
- 可选：加入 **CoT Prompt**（e.g., "Let's think step by step."）
- 训练方式：LoRA + `Trainer` 微调
- 保存 checkpoint：包括模型权重与 tokenizer
- 训练配置：
- Epochs、Batch size、Learning rate
- 最大输入长度 `max_seq_len`

---

## 📊 评估与分析

| 模块 | 工具/方法 |
|------|------------|
| 预测函数 | `greedy decode` 或 `argmax softmax` |
| 主指标 | Accuracy（准确率） |
| 辅助指标 | Precision / Recall / F1（可选） |
| 错误分析 | 查看错误样本偏向、模式分析 |
| 对比评估 | 原始模型 vs 微调模型 |

---

## ✨ 项目亮点与扩展方向

- 🔍 **CoT Prompt 消融实验**：加入与不加 `"Let's think step by step."` 比较效果差异
- ⚙️ **微调策略对比**：LoRA vs Prompt Tuning vs BitFit
- 🧩 **Adapter Tuning 实验**：不同 PEFT 方法在准确率、训练时间、参数量上的比较

---

