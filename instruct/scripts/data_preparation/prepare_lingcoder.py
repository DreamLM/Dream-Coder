import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


def separate_prompt_and_response(row):
    messages = row["messages"]
    assert messages[-1]["role"].lower() == "assistant", messages[-1]
    row["prompt"] = messages[:-1]
    for i, m in enumerate(row["prompt"]):
        row["prompt"][i] = {
            "role": (
                row["prompt"][i]["role"].lower()
                if row["prompt"][i]["role"].lower() != "human"
                else "user"
            ),
            "content": row["prompt"][i]["content"],
        }

    row["response"] = messages[-1]["content"]
    return row


def cleanup_instruction(instruction):
    # Case: weird HTML tags
    instruction = instruction.replace("&lt;p&gt;", "").strip()

    # Case: few-shot
    if instruction.endswith("### Response"):
        instruction = instruction[: -len("### Response")].strip()

        instruction_splited = instruction.split("## Example")
        instruction = (
            instruction_splited[0]
            + "### Instruction"
            + instruction_splited[-1].split("### Instruction")[-1]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/lingcoder")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")

    args = parser.parse_args()

    dataset = datasets.load_dataset("inclusionAI/Ling-Coder-SFT", split="train")
    dataset = dataset.filter(lambda x: x["messages"][-1]["role"].lower() == "assistant")
    dataset = dataset.map(separate_prompt_and_response, num_proc=os.cpu_count() // 2)
    dataset = dataset.remove_columns(["messages"])

    # filter out too long samples
    if args.max_length is not None:
        from transformers import AutoTokenizer
        import os

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        def compute_length(examples):
            lengths = []
            for prompt, response in zip(examples["prompt"], examples["response"]):
                messages = prompt + [{"role": "assistant", "content": response}]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)
                lengths.append(len(tokens))
            return {"length": lengths}

        # Add lengths as a column with batch processing and parallelization
        dataset = dataset.map(
            compute_length,
            batched=True,
            batch_size=64,  # Adjust based on your memory
            num_proc=os.cpu_count() // 2,  # Use multiple cores
            desc="Computing token lengths",
        )

        # Filter based on the precomputed length
        dataset = dataset.filter(lambda x: x["length"] <= args.max_length)
        # Optional: remove the length column to save memory
        dataset = dataset.remove_columns(["length"])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # split train and val
    dataset = dataset.train_test_split(test_size=1000)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
