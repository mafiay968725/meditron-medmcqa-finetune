import json
from datasets import Dataset


def preprocess_medmcqa_train(input_file: str):
    label_map = {1: "A", 2: "B", 3: "C", 4: "D"}

    processed_data = {
        "prompt": [],
        "label": [],
        "question": [],
        "opa": [],
        "opb": [],
        "opc": [],
        "opd": [],
        "cop": [],
        "exp": [],
        "choice_type": [],
        "topic_name": [],
        "id": [],
    }

    raw_data = []
    # 按行读取 JSON Lines 格式的文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                raw_data.append(json.loads(line))

    for item in raw_data:
        question = item.get("question", "").strip()
        if not question:
            continue

        cop = item.get("cop", None)
        if cop not in label_map:
            continue

        # 获取选择类型（single 或 multi）
        choice_type = item.get("choice_type", "single")

        # 获取话题名称
        topic_name = item.get("topic_name", "")

        # 构造 Prompt
        prompt = (
            f"Question: {question}\n"
            f"Options:\n"
            f"A) {item.get('opa', '')}\n"
            f"B) {item.get('opb', '')}\n"
            f"C) {item.get('opc', '')}\n"
            f"D) {item.get('opd', '')}\n"
            f"Choice Type: {choice_type}\n"
            f"Topic: {topic_name}\n"
            f"Answer:"
        )

        label = label_map[cop]

        # 添加到 processed_data 字典
        processed_data["prompt"].append(prompt)
        processed_data["label"].append(label)
        processed_data["question"].append(question)
        processed_data["opa"].append(item.get("opa", ""))
        processed_data["opb"].append(item.get("opb", ""))
        processed_data["opc"].append(item.get("opc", ""))
        processed_data["opd"].append(item.get("opd", ""))
        processed_data["cop"].append(cop)
        processed_data["exp"].append(item.get("exp", ""))
        processed_data["choice_type"].append(choice_type)
        processed_data["topic_name"].append(topic_name)
        processed_data["id"].append(item.get("id", ""))

    dataset = Dataset.from_dict(processed_data)
    return dataset


if __name__ == "__main__":
    train_file = r"D:\自然语言处理\meditron-medmcqa-project\data\train.json"
    processed_train = preprocess_medmcqa_train(train_file)
    print(processed_train)
    print(processed_train[0])
if __name__ == "__main__":
    dev_file = r"D:\自然语言处理\meditron-medmcqa-project\data\dev.json"
    processed_dev = preprocess_medmcqa_train(dev_file)
    print(processed_dev)
    print(processed_dev[0])


def preprocess_medmcqa_test(input_file: str):
    processed_data = {
        "prompt": [],
        "question": [],
        "opa": [],
        "opb": [],
        "opc": [],
        "opd": [],
        "choice_type": [],
        "topic_name": [],
        "id": [],
    }

    raw_data = []
    # 读取 JSON Lines 格式文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    for item in raw_data:
        question = item.get("question", "").strip()
        if not question:
            continue

        # 获取各项字段
        opa = item.get("opa", "")
        opb = item.get("opb", "")
        opc = item.get("opc", "")
        opd = item.get("opd", "")
        choice_type = item.get("choice_type", "single")
        topic_name = item.get("topic_name", "Unknown") or "Unknown"  # 处理 null
        qid = item.get("id", "")

        # 构造 prompt（不含答案）
        prompt = (
            f"Question: {question}\n"
            f"Options:\n"
            f"A) {opa}\n"
            f"B) {opb}\n"
            f"C) {opc}\n"
            f"D) {opd}\n"
            f"Choice Type: {choice_type}\n"
            f"Topic: {topic_name}"
        )

        processed_data["prompt"].append(prompt)
        processed_data["question"].append(question)
        processed_data["opa"].append(opa)
        processed_data["opb"].append(opb)
        processed_data["opc"].append(opc)
        processed_data["opd"].append(opd)
        processed_data["choice_type"].append(choice_type)
        processed_data["topic_name"].append(topic_name)
        processed_data["id"].append(qid)

    dataset = Dataset.from_dict(processed_data)
    return dataset
if __name__ == "__main__":
    test_file = r"D:\自然语言处理\meditron-medmcqa-project\data\test.json"
    processed_test = preprocess_medmcqa_test(test_file)
    print(processed_test)
    print(processed_test[0]["prompt"])