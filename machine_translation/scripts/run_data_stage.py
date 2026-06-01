from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from akkadian_mt.data import make_dev_split
from inspect_data import summarize_csv


REQUIRED_FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def require_kaggle_files(raw_dir: Path) -> dict[str, Path]:
    paths = {name: raw_dir / name for name in REQUIRED_FILES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Kaggle files are missing. Download them from the competition Data tab "
            "or run ./scripts/download_kaggle.sh after configuring Kaggle CLI:\n"
            f"  - {joined}"
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Kaggle data and create dev splits.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--dev-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    try:
        paths = require_kaggle_files(raw_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print("Inspecting Kaggle files...")
    for path in paths.values():
        summarize_csv(path, sample_size=args.sample_size)

    print("\nCreating raw train/dev split...")
    raw_train, raw_dev = make_dev_split(
        dataset_path=paths["train.csv"],
        output_dir=output_dir / "raw",
        dev_size=args.dev_size,
        seed=args.seed,
        normalize=False,
    )
    print(f"raw train: {raw_train}")
    print(f"raw dev:   {raw_dev}")

    print("\nCreating normalized train/dev split...")
    norm_train, norm_dev = make_dev_split(
        dataset_path=paths["train.csv"],
        output_dir=output_dir / "normalized",
        dev_size=args.dev_size,
        seed=args.seed,
        normalize=True,
    )
    print(f"normalized train: {norm_train}")
    print(f"normalized dev:   {norm_dev}")


if __name__ == "__main__":
    main()
