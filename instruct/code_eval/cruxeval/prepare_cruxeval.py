"""https://github.com/facebookresearch/cruxeval/blob/main/prompts.py"""

import argparse
import json
import os

from datasets import load_dataset


def make_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""


def make_direct_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""


def make_direct_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""


def make_cot_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17.

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16.
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""


def load_cruxeval_data():
    """Load CRUXEval dataset from jsonl file"""
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    data = ds.to_list()
    return data


def create_prompt_entry(prompt_content, sample_id, prompt_type, include_system=False):
    """Create a single prompt entry in the required format"""
    messages = []

    if include_system:
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in Python code analysis and execution.",
            }
        )

    messages.append({"role": "user", "content": prompt_content})

    return {
        "messages": messages,
        "id": sample_id,
        "prompt_type": prompt_type,
        "sample_id": sample_id.split("_")[-1] if "_" in sample_id else sample_id,
    }


def generate_all_prompts(data, include_system=False):
    """Generate all prompts for the dataset"""
    all_prompts = []
    prompt_id = 1

    for sample in data:
        code = sample["code"]
        input_data = sample["input"]
        output_data = sample["output"]
        sample_base_id = sample["id"]

        # Output prediction prompts
        # 1. Direct output prompt
        prompt_content = make_direct_output_prompt((code, input_data))
        all_prompts.append(
            create_prompt_entry(
                prompt_content,
                f"{sample_base_id}_output_direct",
                "output_prediction_direct",
                include_system,
            )
        )

        # 2. CoT output prompt
        prompt_content = make_cot_output_prompt((code, input_data))
        all_prompts.append(
            create_prompt_entry(
                prompt_content,
                f"{sample_base_id}_output_cot",
                "output_prediction_cot",
                include_system,
            )
        )

        # Input prediction prompts
        # 3. Direct input prompt
        prompt_content = make_direct_input_prompt((code, output_data))
        all_prompts.append(
            create_prompt_entry(
                prompt_content,
                f"{sample_base_id}_input_direct",
                "input_prediction_direct",
                include_system,
            )
        )

        # 4. CoT input prompt
        prompt_content = make_cot_input_prompt((code, output_data))
        all_prompts.append(
            create_prompt_entry(
                prompt_content,
                f"{sample_base_id}_input_cot",
                "input_prediction_cot",
                include_system,
            )
        )

    return all_prompts


def save_prompts_to_jsonl(prompts, output_path):
    """Save prompts to jsonl file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(prompts)} prompts to {output_path}")


def main(args):
    # Handle system message setting
    include_system = args.include_system

    # Load data
    data = load_cruxeval_data()

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Loaded {len(data)} samples")

    # Generate prompts
    print("Generating prompts...")
    if args.prompt_types:
        # Generate only specific prompt types
        all_prompts = []
        prompt_id = 1

        for sample in data:
            code = sample["code"]
            input_data = sample["input"]
            output_data = sample["output"]
            sample_base_id = sample["id"]

            if "output_direct" in args.prompt_types:
                prompt_content = make_direct_output_prompt((code, input_data))
                all_prompts.append(
                    create_prompt_entry(
                        prompt_content, sample_base_id, "output_direct", include_system
                    )
                )

            if "output_cot" in args.prompt_types:
                prompt_content = make_cot_output_prompt((code, input_data))
                all_prompts.append(
                    create_prompt_entry(
                        prompt_content, sample_base_id, "output_cot", include_system
                    )
                )

            if "input_direct" in args.prompt_types:
                prompt_content = make_direct_input_prompt((code, output_data))
                all_prompts.append(
                    create_prompt_entry(
                        prompt_content, sample_base_id, "input_direct", include_system
                    )
                )

            if "input_cot" in args.prompt_types:
                prompt_content = make_cot_input_prompt((code, output_data))
                all_prompts.append(
                    create_prompt_entry(
                        prompt_content, sample_base_id, "input_cot", include_system
                    )
                )
    else:
        # Generate all prompts
        all_prompts = generate_all_prompts(data, include_system)

    # Save to file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving prompts to {args.output_path}...")
    save_prompts_to_jsonl(all_prompts, args.output_path)

    # Print statistics
    prompt_type_counts = {}
    for prompt in all_prompts:
        prompt_type = prompt["prompt_type"]
        prompt_type_counts[prompt_type] = prompt_type_counts.get(prompt_type, 0) + 1

    print("\nPrompt type statistics:")
    for prompt_type, count in prompt_type_counts.items():
        print(f"  {prompt_type}: {count}")

    print(f"\nTotal prompts generated: {len(all_prompts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CRUXEval prompts in jsonl format"
    )
    parser.add_argument(
        "--output_path",
        default="data/cruxeval/test.jsonl",
        help="Output path for generated prompts",
    )
    parser.add_argument(
        "--include_system",
        action="store_true",
        help="Include system message in prompts",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--prompt_types",
        nargs="+",
        choices=["output_direct", "output_cot", "input_direct", "input_cot"],
        help="Specific prompt types to generate",
    )

    args = parser.parse_args()

    main(args)
