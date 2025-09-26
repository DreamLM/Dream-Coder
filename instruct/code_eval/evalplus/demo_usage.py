#!/usr/bin/env python3
"""
演示如何使用生成的instruct格式评估数据
"""

import json
from typing import List, Dict, Any


def load_instruct_data(jsonl_file: str) -> List[Dict[str, Any]]:
    """加载instruct格式的数据"""
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_for_generation(messages: List[Dict[str, str]], tokenizer=None) -> str:
    """
    将messages格式化为用于生成的提示
    如果有tokenizer，使用chat_template；否则使用简单拼接
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        # 使用HuggingFace tokenizer的chat_template
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"使用chat_template失败: {e}")
            # 降级到简单格式

    # 简单的文本拼接格式
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|user|>\n{content}\n\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}"

    # 如果最后不是assistant消息，添加生成提示
    if not messages or messages[-1]["role"] != "assistant":
        formatted += "<|assistant|>\n"

    return formatted


def demo_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 加载数据
    try:
        data = load_instruct_data("humaneval_instruct.jsonl")
        print(f"加载了 {len(data)} 个任务")

        # 显示第一个任务
        first_task = data[0]
        print(f"\n任务ID: {first_task['task_id']}")
        print(f"数据集: {first_task['dataset']}")
        print(f"函数入口点: {first_task['entry_point']}")
        print(f"模板: {first_task['instruction_template']}")

        # 显示messages
        print(f"\nMessages格式:")
        for i, msg in enumerate(first_task["messages"]):
            print(f"  Message {i+1} ({msg['role']}):")
            content = msg["content"]
            if len(content) > 200:
                print(f"    {content[:200]}...")
            else:
                print(f"    {content}")

    except FileNotFoundError:
        print("请先运行 prepare_instruct_prompts.py 生成数据文件")


def demo_generation_format():
    """演示生成格式"""
    print("\n=== 生成格式示例 ===")

    try:
        data = load_instruct_data("humaneval_instruct.jsonl")
        first_task = data[0]

        # 格式化为生成提示
        generation_prompt = format_for_generation(first_task["messages"])

        print("用于生成的格式化提示:")
        print("-" * 50)
        print(generation_prompt)
        print("-" * 50)

    except FileNotFoundError:
        print("请先运行 prepare_instruct_prompts.py 生成数据文件")


def demo_with_transformers():
    """演示与transformers库的集成"""
    print("\n=== Transformers集成示例 ===")

    try:
        # 这里只是示例代码，实际使用时需要安装transformers
        example_code = """
# 使用transformers的示例代码:
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# 加载数据
data = load_instruct_data("humaneval_instruct.jsonl")

# 格式化单个任务
task = data[0]
formatted_prompt = tokenizer.apply_chat_template(
    task["messages"],
    tokenize=False,
    add_generation_prompt=True
)

# 生成代码
inputs = tokenizer(formatted_prompt, return_tensors="pt")
# ... 模型推理代码 ...
"""
        print(example_code)

    except Exception as e:
        print(f"演示代码: {e}")


def compare_templates():
    """比较不同模板的差异"""
    print("\n=== 模板比较示例 ===")

    files_to_compare = [
        ("humaneval_instruct.jsonl", "default"),
        ("instruct_chinese.jsonl", "chinese"),
        ("instruct_perf.jsonl", "perf-instruct"),
    ]

    for filename, template_name in files_to_compare:
        try:
            data = load_instruct_data(filename)
            if data:
                first_task = data[0]
                user_msg = first_task["messages"][0]["content"]

                print(f"\n{template_name} 模板:")
                # 只显示instruction部分（第一行）
                instruction = user_msg.split("\n")[0]
                print(f"  {instruction}")

        except FileNotFoundError:
            print(f"  {filename} 不存在")


def main():
    """主函数"""
    print("EvalPlus Instruct格式数据使用演示")
    print("=" * 50)

    demo_basic_usage()
    demo_generation_format()
    demo_with_transformers()
    compare_templates()

    print("\n=== 使用建议 ===")
    print("1. 对于标准代码生成任务，使用 'default' 模板")
    print("2. 对于性能敏感任务，使用 'perf-instruct' 或 'perf-CoT' 模板")
    print("3. 对于中文模型，使用 'chinese' 或 'chinese-perf' 模板")
    print("4. 使用 --include-response 可以获得更完整的对话格式")
    print("5. 配合transformers的apply_chat_template使用效果最佳")


if __name__ == "__main__":
    main()
