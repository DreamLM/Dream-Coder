import argparse

import pandas as pd

from src.postprocess import postprocess_cruxeval, postprocess_multipl_e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    if args.type == "cruxeval":
        postprocess_cruxeval(df, args.output)
    elif args.type == "multipl_e":
        postprocess_multipl_e(df, args.output)
    else:
        raise ValueError(f"Invalid type: {args.type}")
