from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
from pathlib import Path

from akkadian_mt.normalization import normalize_akkadian


SOURCE_CANDIDATES = ["akkadian", "source", "src", "transliteration", "text"]


def infer_source(fieldnames: list[str]) -> str:
    lower_to_original = {name.lower(): name for name in fieldnames}
    source = next((lower_to_original[c] for c in SOURCE_CANDIDATES if c in lower_to_original), None)
    if source is None:
        raise ValueError(f"Cannot infer source column from {fieldnames}")
    return source


def read_sources(path: Path, normalize: bool) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return []
        source_col = infer_source(reader.fieldnames)
        values = [row.get(source_col, "") for row in reader]
    if normalize:
        values = [normalize_akkadian(value) for value in values]
    return [value.strip() for value in values if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Usually Kaggle test.csv")
    parser.add_argument("--candidate", required=True, help="Train/external corpus to check")
    parser.add_argument("--near-threshold", type=float, default=0.96)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-near-reports", type=int, default=20)
    args = parser.parse_args()

    reference = read_sources(Path(args.reference), normalize=args.normalize)
    candidate = read_sources(Path(args.candidate), normalize=args.normalize)

    reference_set = set(reference)
    exact = sorted(set(candidate) & reference_set)
    print(f"reference rows: {len(reference)}")
    print(f"candidate rows: {len(candidate)}")
    print(f"exact overlaps: {len(exact)}")
    for item in exact[:20]:
        print(f"EXACT\t{item}")

    near_reports = 0
    print("near duplicates:")
    # Intended for external corpus audit, not for huge all-pairs jobs.
    for cand in candidate:
        if cand in reference_set:
            continue
        for ref in reference:
            score = SequenceMatcher(None, cand, ref).ratio()
            if score >= args.near_threshold:
                print(f"NEAR\t{score:.4f}\tCAND={cand}\tREF={ref}")
                near_reports += 1
                break
        if near_reports >= args.max_near_reports:
            break
    print(f"near reports shown: {near_reports}")


if __name__ == "__main__":
    main()
