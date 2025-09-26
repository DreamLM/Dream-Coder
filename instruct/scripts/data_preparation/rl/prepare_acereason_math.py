"""https://huggingface.co/datasets/nvidia/AceReason-Math/blob/main/math.jsonl"""

import pandas as pd
from huggingface_hub import hf_hub_download

import json

import numpy as np


if __name__ == "__main__":
    df = pd.read_json(
        hf_hub_download(
            repo_id="nvidia/AceReason-Math", filename="math.jsonl", repo_type="dataset"
        ),
        lines=True,
    )
    df["prompt"] = df.problem.apply(
        lambda x: x.strip()
        + "\n\nLet's think step by step and output the final answer within \\boxed{}."
    )

    def get_gt(row):
        row["info"] = json.dumps(
            {
                "ground_truth": row["answer"],
            }
        )
        return row

    df["task_id"] = np.arange(len(df))
    df["task_id"] = df.apply(lambda x: f"acereason_math/test/{x['task_id']}", axis=1)
    df["data_source"] = "aime24"
    df = df.apply(get_gt, axis=1)
    df[["task_id", "prompt", "data_source", "info"]].to_parquet(
        "/home/ndfl4zki/ndfl4zkiuser04/data/code_rl/acereason_math.parquet"
    )
