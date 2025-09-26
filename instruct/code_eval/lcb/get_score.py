import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--start_date", type=str, default="2024-10-01")
    parser.add_argument("--end_date", type=str, default="2025-05-01")
    args = parser.parse_args()

    df = pd.read_json(args.input)
    df = df[df["contest_date"] >= args.start_date]
    df = df[df["contest_date"] <= args.end_date]

    # get the mean of pass@1
    print(df["pass@1"].mean())

    # get the mean of pass@1 for each difficulty
    print(df.groupby("difficulty")["pass@1"].mean())
