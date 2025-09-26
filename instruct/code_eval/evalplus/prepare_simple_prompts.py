#!/usr/bin/env python3
"""
简化版本：只准备最基本的提示字段 (task_id, prompt) 用于LLM代码生成
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import get_evalperf_data


def extract_simple_prompt(task: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """提取简化的提示格式，只包含generation需要的字段"""
    return {
        "task_id": task["task_id"],
        "dataset": dataset_name,
        "prompt": task["prompt"],
        "entry_point": task["entry_point"],
    }


def collect_simple_prompts(
    include_humaneval: bool = True,
    include_mbpp: bool = True,
    include_evalperf: bool = True,
    mini: bool = False,
) -> List[Dict[str, Any]]:
    """收集简化的提示数据"""
    all_prompts = []

    if include_humaneval:
        print("正在加载 HumanEval+ 数据集...")
        try:
            humaneval_data = get_human_eval_plus(mini=mini)
            for task_id, task in humaneval_data.items():
                all_prompts.append(extract_simple_prompt(task, "humaneval_plus"))
            print(f"已添加 {len(humaneval_data)} 个 HumanEval+ 任务")
        except Exception as e:
            print(f"加载 HumanEval+ 时出错: {e}")

    if include_mbpp:
        print("正在加载 MBPP+ 数据集...")
        try:
            mbpp_data = get_mbpp_plus(mini=mini)
            for task_id, task in mbpp_data.items():
                all_prompts.append(extract_simple_prompt(task, "mbpp_plus"))
            print(f"已添加 {len(mbpp_data)} 个 MBPP+ 任务")
        except Exception as e:
            print(f"加载 MBPP+ 时出错: {e}")

    if include_evalperf:
        print("正在加载 EvalPerf 数据集...")
        try:
            evalperf_data = get_evalperf_data()
            for task_id, task in evalperf_data.items():
                all_prompts.append(extract_simple_prompt(task, "evalperf"))
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

    print(f"已将 {len(prompts)} 个提示保存到 {output_file}")


def print_summary(prompts: List[Dict[str, Any]]):
    """打印数据集摘要"""
    print("\n=== 简化提示数据集摘要 ===")

    # 按数据集统计
    dataset_counts = {}
    for prompt in prompts:
        dataset = prompt["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print("各数据集任务数量:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} 个任务")

    print(f"总计: {len(prompts)} 个任务")

    # 显示示例
    if prompts:
        print(f"\n示例任务 (来自 {prompts[0]['dataset']}):")
        example = prompts[0]
        print(f"  task_id: {example['task_id']}")
        print(f"  entry_point: {example['entry_point']}")
        print(f"  prompt 预览: {example['prompt'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="准备简化的评估提示为JSONL文件")
    parser.add_argument(
        "--output",
        "-o",
        default="simple_eval_prompts.jsonl",
        help="输出JSONL文件路径 (默认: simple_eval_prompts.jsonl)",
    )
    parser.add_argument(
        "--no-humaneval", action="store_true", help="不包含HumanEval+数据集"
    )
    parser.add_argument("--no-mbpp", action="store_true", help="不包含MBPP+数据集")
    parser.add_argument(
        "--no-evalperf", action="store_true", help="不包含EvalPerf数据集"
    )
    parser.add_argument(
        "--mini", action="store_true", help="使用mini版本的数据集 (更快的评估)"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="只显示摘要信息，不保存文件"
    )
    parser.add_argument(
        "--humaneval-only", action="store_true", help="只包含HumanEval+数据集"
    )
    parser.add_argument("--mbpp-only", action="store_true", help="只包含MBPP+数据集")

    args = parser.parse_args()

    # 处理互斥选项
    if args.humaneval_only:
        include_humaneval, include_mbpp, include_evalperf = True, False, False
    elif args.mbpp_only:
        include_humaneval, include_mbpp, include_evalperf = False, True, False
    else:
        include_humaneval = not args.no_humaneval
        include_mbpp = not args.no_mbpp
        include_evalperf = not args.no_evalperf

    # 收集所有提示
    prompts = collect_simple_prompts(
        include_humaneval=include_humaneval,
        include_mbpp=include_mbpp,
        include_evalperf=include_evalperf,
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
        print(f"\n完成! 简化的评估提示已保存到 {args.output}")

        # 显示文件格式说明
        print("\n文件格式说明:")
        print("每行包含一个JSON对象，包含以下字段:")
        print("  - task_id: 任务标识符")
        print("  - dataset: 数据集名称 (humaneval_plus/mbpp_plus/evalperf)")
        print("  - prompt: 代码生成提示")
        print("  - entry_point: 函数入口点名称")
    else:
        print("\n摘要模式: 未保存文件")


if __name__ == "__main__":
    main()
