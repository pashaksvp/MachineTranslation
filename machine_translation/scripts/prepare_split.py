from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akkadian_mt.data import make_dev_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--dev-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    make_dev_split(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        dev_size=args.dev_size,
        seed=args.seed,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
