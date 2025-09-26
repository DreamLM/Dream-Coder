"""https://huggingface.co/datasets/nvidia/AceReason-Math/blob/main/math.jsonl"""

import pandas as pd
from huggingface_hub import hf_hub_download

import json

import numpy as np


if __name__ == "__main__":
    df = pd.read_parquet(
        hf_hub_download(
            repo_id="LLM360/guru-RL-92k",
            filename="train/simulation__codeio_3.7k.parquet",
            repo_type="dataset",
        )
    )

    def get_gt(row):
        row["info"] = json.dumps(
            {
                "ground_truth": row["reward_model"]["ground_truth"],
            }
        )
        return row

    df["task_id"] = np.arange(len(df))
    df["task_id"] = df.apply(lambda x: f"guru_codeio/test/{x['task_id']}", axis=1)
    df["data_source"] = "codeio"
    df = df.apply(get_gt, axis=1)
    df["difficulty_level"] = df["qwen2.5_7b_pass_rate"]
    df[["task_id", "prompt", "data_source", "info", "difficulty_level"]].to_parquet(
        "/home/ndfl4zki/ndfl4zkiuser04/data/code_rl/guru_codeio.parquet"
    )

    # filtered version
    df = df[df["qwen3_30b_pass_rate"] < 1]
    df[["task_id", "prompt", "data_source", "info"]].to_parquet(
        "/home/ndfl4zki/ndfl4zkiuser04/data/code_rl/guru_codeio_filtered.parquet"
    )

    df_hard = df[df["qwen3_30b_pass_rate"] == 0]
    df = df[df["qwen3_30b_pass_rate"] > 0]
    df_hard = df_hard.sample(frac=0.25)
    df = pd.concat([df, df_hard])
    df[["task_id", "prompt", "data_source", "info"]].to_parquet(
        "/home/ndfl4zki/ndfl4zkiuser04/data/code_rl/guru_codeio_filteredv4.parquet"
    )
