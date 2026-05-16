from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


SOURCE_CANDIDATES = ["akkadian", "source", "src", "transliteration", "text"]
TARGET_CANDIDATES = ["english", "target", "translation", "tgt"]


def infer_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {name.lower(): name for name in fieldnames}
    return next((lower_to_original[c] for c in candidates if c in lower_to_original), None)


def summarize_csv(path: Path, sample_size: int = 3) -> None:
    print(f"\n== {path} ==")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            print("Empty CSV or missing header")
            return

        fieldnames = reader.fieldnames
        source_col = infer_column(fieldnames, SOURCE_CANDIDATES)
        target_col = infer_column(fieldnames, TARGET_CANDIDATES)
        print(f"columns: {fieldnames}")
        print(f"inferred source: {source_col}")
        print(f"inferred target: {target_col}")

        rows = 0
        missing = Counter()
        source_values: list[str] = []
        target_values: list[str] = []
        samples: list[dict[str, str]] = []

        for row in reader:
            rows += 1
            if len(samples) < sample_size:
                samples.append(row)
            for col in fieldnames:
                if not row.get(col):
                    missing[col] += 1
            if source_col:
                source_values.append(row.get(source_col, ""))
            if target_col:
                target_values.append(row.get(target_col, ""))

    print(f"rows: {rows}")
    if missing:
        print(f"missing values: {dict(missing)}")
    if source_values:
        unique_sources = len(set(source_values))
        duplicate_sources = len(source_values) - unique_sources
        avg_source_len = sum(len(x) for x in source_values) / max(len(source_values), 1)
        print(f"unique sources: {unique_sources}")
        print(f"duplicate sources: {duplicate_sources}")
        print(f"avg source chars: {avg_source_len:.2f}")
    if target_values:
        unique_targets = len(set(target_values))
        duplicate_targets = len(target_values) - unique_targets
        avg_target_len = sum(len(x) for x in target_values) / max(len(target_values), 1)
        print(f"unique targets: {unique_targets}")
        print(f"duplicate targets: {duplicate_targets}")
        print(f"avg target chars: {avg_target_len:.2f}")

    print("samples:")
    for sample in samples:
        compact = {k: sample[k] for k in fieldnames[: min(len(fieldnames), 4)]}
        print(compact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="CSV files to inspect")
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()

    for path in args.paths:
        summarize_csv(Path(path), sample_size=args.sample_size)


if __name__ == "__main__":
    main()
