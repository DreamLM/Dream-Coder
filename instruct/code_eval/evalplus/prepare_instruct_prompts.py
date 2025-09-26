#!/usr/bin/env python3
"""
为instruct模型准备HuggingFace messages格式的评估提示
包含 HumanEval+, MBPP+, 和 EvalPerf 数据集
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import get_evalperf_data


# EvalPlus中定义的instruction prefix模板
INSTRUCTION_TEMPLATES = {
    "default": "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
    "perf-instruct": "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:",
    "perf-CoT": "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:",
    "chinese": "请提供一个完整的Python脚本来解决以下问题，并将代码放在markdown代码块中：",
    "chinese-perf": "请提供一个高效的、完整的Python脚本来解决以下问题，并将代码放在markdown代码块中：",
}

RESPONSE_TEMPLATES = {
    "default": "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
    "perf": "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:",
    "chinese": "以下是一个完整的Python脚本，包含解决问题并通过相应测试的函数：",
    "chinese-perf": "以下是一个高效的、完整的Python脚本，包含解决问题并通过相应测试的函数：",
}


def format_instruct_prompt(
    task: Dict[str, Any],
    dataset_name: str,
    instruction_template: str = "default",
    include_response: bool = False,
    custom_instruction: str = None,
) -> Dict[str, Any]:
    """将任务格式化为HuggingFace messages格式"""

    # 获取instruction prefix
    if custom_instruction:
        instruction_prefix = custom_instruction
    else:
        instruction_prefix = INSTRUCTION_TEMPLATES.get(
            instruction_template, INSTRUCTION_TEMPLATES["default"]
        )

    # 构建用户消息内容
    user_content = f"""{instruction_prefix}
```python
{task['prompt'].strip()}
```"""

    # 基本的messages格式
    messages = [{"role": "user", "content": user_content}]

    # 如果包含response template，添加assistant消息的开头
    if include_response:
        response_key = "perf" if "perf" in instruction_template else "default"
        if "chinese" in instruction_template:
            response_key = (
                "chinese-perf" if "perf" in instruction_template else "chinese"
            )

        response_prefix = RESPONSE_TEMPLATES.get(
            response_key, RESPONSE_TEMPLATES["default"]
        )
        assistant_content = f"""{response_prefix}
```python
"""
        messages.append({"role": "assistant", "content": assistant_content})

    return {
        "task_id": task["task_id"],
        "dataset": dataset_name,
        "messages": messages,
        "entry_point": task["entry_point"],
        "instruction_template": instruction_template,
        "metadata": {
            "original_prompt": task["prompt"],
            "has_response_template": include_response,
        },
    }


def collect_instruct_prompts(
    include_humaneval: bool = True,
    include_mbpp: bool = True,
    include_evalperf: bool = True,
    instruction_template: str = "default",
    include_response: bool = False,
    custom_instruction: str = None,
    mini: bool = False,
) -> List[Dict[str, Any]]:
    """收集instruct格式的提示数据"""
    all_prompts = []

    if include_humaneval:
        print("正在加载 HumanEval+ 数据集...")
        try:
            humaneval_data = get_human_eval_plus(mini=mini)
            for task_id, task in humaneval_data.items():
                prompt_data = format_instruct_prompt(
                    task,
                    "humaneval_plus",
                    instruction_template,
                    include_response,
                    custom_instruction,
                )
                all_prompts.append(prompt_data)
            print(f"已添加 {len(humaneval_data)} 个 HumanEval+ 任务")
        except Exception as e:
            print(f"加载 HumanEval+ 时出错: {e}")

    if include_mbpp:
        print("正在加载 MBPP+ 数据集...")
        try:
            mbpp_data = get_mbpp_plus(mini=mini)
            for task_id, task in mbpp_data.items():
                prompt_data = format_instruct_prompt(
                    task,
                    "mbpp_plus",
                    instruction_template,
                    include_response,
                    custom_instruction,
                )
                all_prompts.append(prompt_data)
            print(f"已添加 {len(mbpp_data)} 个 MBPP+ 任务")
        except Exception as e:
            print(f"加载 MBPP+ 时出错: {e}")

    if include_evalperf:
        print("正在加载 EvalPerf 数据集...")
        try:
            evalperf_data = get_evalperf_data()
            # EvalPerf建议使用perf-instruct模板
            evalperf_template = instruction_template
            if instruction_template == "default":
                evalperf_template = "perf-instruct"

            for task_id, task in evalperf_data.items():
                prompt_data = format_instruct_prompt(
                    task,
                    "evalperf",
                    evalperf_template,
                    include_response,
                    custom_instruction,
                )
                all_prompts.append(prompt_data)
            print(f"已添加 {len(evalperf_data)} 个 EvalPerf 任务")
        except Exception as e:
            print(f"加载 EvalPerf 时出错: {e}")

    return all_prompts


def save_to_jsonl(prompts: List[Dict[str, Any]], output_file: str):
    """将提示保存为JSONL格式"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

    print(f"已将 {len(prompts)} 个instruct提示保存到 {output_file}")


def print_summary(prompts: List[Dict[str, Any]]):
    """打印数据集摘要"""
    print("\n=== Instruct模型提示数据集摘要 ===")

    # 按数据集统计
    dataset_counts = {}
    template_counts = {}
    for prompt in prompts:
        dataset = prompt["dataset"]
        template = prompt["instruction_template"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        template_counts[template] = template_counts.get(template, 0) + 1

    print("各数据集任务数量:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} 个任务")

    print(f"\n使用的instruction模板:")
    for template, count in template_counts.items():
        print(f"  {template}: {count} 个任务")

    print(f"\n总计: {len(prompts)} 个任务")

    # 显示示例
    if prompts:
        print(f"\n示例任务 (来自 {prompts[0]['dataset']}):")
        example = prompts[0]
        print(f"  task_id: {example['task_id']}")
        print(f"  entry_point: {example['entry_point']}")
        print(f"  instruction_template: {example['instruction_template']}")
        print(f"  messages:")
        for msg in example["messages"]:
            print(
                f"    {msg['role']}: {msg['content'][:100]}..."
                if len(msg["content"]) > 100
                else f"    {msg['role']}: {msg['content']}"
            )


def show_templates():
    """显示可用的instruction模板"""
    print("=== 可用的Instruction模板 ===")
    for key, template in INSTRUCTION_TEMPLATES.items():
        print(f"\n{key}:")
        print(f"  {template}")


def main():
    parser = argparse.ArgumentParser(
        description="为instruct模型准备HuggingFace messages格式的评估提示"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="instruct_eval_prompts.jsonl",
        help="输出JSONL文件路径 (默认: instruct_eval_prompts.jsonl)",
    )

    # 数据集选择
    parser.add_argument(
        "--no-humaneval", action="store_true", help="不包含HumanEval+数据集"
    )
    parser.add_argument("--no-mbpp", action="store_true", help="不包含MBPP+数据集")
    parser.add_argument(
        "--no-evalperf", action="store_true", help="不包含EvalPerf数据集"
    )
    parser.add_argument(
        "--humaneval-only", action="store_true", help="只包含HumanEval+数据集"
    )
    parser.add_argument("--mbpp-only", action="store_true", help="只包含MBPP+数据集")
    parser.add_argument(
        "--evalperf-only", action="store_true", help="只包含EvalPerf数据集"
    )

    # 模板和格式选项
    parser.add_argument(
        "--template",
        "-t",
        default="default",
        choices=list(INSTRUCTION_TEMPLATES.keys()),
        help="选择instruction模板",
    )
    parser.add_argument("--custom-instruction", help="使用自定义的instruction prefix")
    parser.add_argument(
        "--include-response",
        action="store_true",
        help="包含assistant response的开头模板",
    )
    parser.add_argument(
        "--show-templates", action="store_true", help="显示所有可用的模板并退出"
    )

    # 其他选项
    parser.add_argument("--mini", action="store_true", help="使用mini版本数据集")
    parser.add_argument(
        "--summary-only", action="store_true", help="只显示摘要信息，不保存文件"
    )

    args = parser.parse_args()

    if args.show_templates:
        show_templates()
        return

    # 处理互斥选项
    if args.humaneval_only:
        include_humaneval, include_mbpp, include_evalperf = True, False, False
    elif args.mbpp_only:
        include_humaneval, include_mbpp, include_evalperf = False, True, False
    elif args.evalperf_only:
        include_humaneval, include_mbpp, include_evalperf = False, False, True
    else:
        include_humaneval = not args.no_humaneval
        include_mbpp = not args.no_mbpp
        include_evalperf = not args.no_evalperf

    # 收集所有提示
    prompts = collect_instruct_prompts(
        include_humaneval=include_humaneval,
        include_mbpp=include_mbpp,
        include_evalperf=include_evalperf,
        instruction_template=args.template,
        include_response=args.include_response,
        custom_instruction=args.custom_instruction,
        mini=args.mini,
    )

    if not prompts:
        print("错误: 没有收集到任何提示数据")
        return

    # 打印摘要
    print_summary(prompts)

    # 保存文件
    if not args.summary_only:
        save_to_jsonl(prompts, args.output)
        print(f"\n完成! Instruct模型提示已保存到 {args.output}")

        # 显示文件格式说明
        print(f"\n文件格式说明:")
        print("每行包含一个JSON对象，包含以下字段:")
        print("  - task_id: 任务标识符")
        print("  - dataset: 数据集名称")
        print("  - messages: HuggingFace chat messages格式")
        print("  - entry_point: 函数入口点名称")
        print("  - instruction_template: 使用的模板名称")
        print("  - metadata: 额外的元数据信息")

        print(f"\n使用示例:")
        print("# 配合transformers使用:")
        print(
            "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)"
        )
    else:
        print("\n摘要模式: 未保存文件")


if __name__ == "__main__":
    main()
