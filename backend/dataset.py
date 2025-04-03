from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K")
dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% eval

def format_as_chat(example):
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
    }

chat_dataset = dataset.map(format_as_chat, remove_columns=["instruction", "input", "output"])
chat_dataset["train"].to_json("python_chat_train.jsonl", orient="records", lines=True)
chat_dataset["test"].to_json("python_chat_eval.jsonl", orient="records", lines=True)
