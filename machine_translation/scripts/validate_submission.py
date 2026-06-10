from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def validate_submission(sample_path: str | Path, submission_path: str | Path) -> list[str]:
    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    errors: list[str] = []

    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"columns mismatch: expected {list(sample.columns)}, got {list(submission.columns)}"
        )
        return errors

    id_col = sample.columns[0]
    prediction_col = sample.columns[1]

    if len(submission) != len(sample):
        errors.append(f"row count mismatch: expected {len(sample)}, got {len(submission)}")

    if submission[id_col].duplicated().any():
        duplicate_ids = submission.loc[submission[id_col].duplicated(), id_col].tolist()
        errors.append(f"duplicate ids: {duplicate_ids[:10]}")

    if len(submission) == len(sample) and not submission[id_col].equals(sample[id_col]):
        errors.append("id values/order do not match sample_submission")

    if submission[prediction_col].isna().any():
        errors.append(f"empty predictions in column={prediction_col!r}")

    blank_mask = submission[prediction_col].astype(str).str.strip().eq("")
    if blank_mask.any():
        blank_ids = submission.loc[blank_mask, id_col].tolist()
        errors.append(f"blank predictions for ids: {blank_ids[:10]}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Kaggle submission CSV")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--submission", default="data/results.csv")
    args = parser.parse_args()

    errors = validate_submission(args.sample, args.submission)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Submission validation passed")


if __name__ == "__main__":
    main()

