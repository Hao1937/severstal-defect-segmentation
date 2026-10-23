"""Create a shared 80/20 multilabel split and summarize the downloaded data."""

import argparse
import json
from pathlib import Path

from .data import make_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/severstal")
    parser.add_argument("--output", default="data/split_seed42.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit("Split already exists; reuse it rather than silently replacing it")
    split = make_split(args.data_dir, args.output)
    print(json.dumps(split["counts"], indent=2))


if __name__ == "__main__":
    main()
