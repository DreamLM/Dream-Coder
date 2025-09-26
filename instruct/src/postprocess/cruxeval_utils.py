"""https://github.com/facebookresearch/cruxeval/blob/main/openai/openai_prompt.py

The generated responses should be postprocessed to the following format:
{
    "sample_0": ["f([1, 1, 1, 1, 3, 3])", "f([])"],
    ...
    "sample_799": ["f('~neqe-;ew22')", "f('~neqe-;ew22')"]
}
"""

import json
from typing import List

import pandas as pd


def extract_answer_direct_output(gen):
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()


def extract_answer_direct_input(gen):
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()


def extract_answer_cot_input(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[0]
        if "assert f" in gen:
            gen = "f" + gen.split("assert f")[1].strip()
        return gen.strip()
    else:
        return gen.split("\n")[-1].strip()


def extract_answer_cot_output(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[1]
        return gen.strip()
    else:
        return gen.split("\n")[-1].strip()


def postprocess_cruxeval(df: pd.DataFrame, output_path: str):
    ids: List[str] = df["id"].tolist()
    generations: List[List[str]] = df["generations"].tolist()
    prompt_types: List[str] = df["prompt_type"].tolist()

    def remove_answer(gen):
        return gen.split("[/ANSWER]")[0].strip()

    output = {}
    for id, gens, type in zip(ids, generations, prompt_types):
        if type == "output_direct":
            output[id] = [
                extract_answer_direct_output(remove_answer(gen)) for gen in gens
            ]
        elif type == "input_direct":
            output[id] = [
                extract_answer_direct_input(remove_answer(gen)) for gen in gens
            ]
        elif type == "input_cot":
            output[id] = [extract_answer_cot_input(remove_answer(gen)) for gen in gens]
        elif type == "output_cot":
            output[id] = [extract_answer_cot_output(remove_answer(gen)) for gen in gens]
        else:
            raise ValueError(f"Invalid type: {type}")

    json.dump(output, open(output_path, "w"), ensure_ascii=False, indent=4)

    return pd.DataFrame(output)
