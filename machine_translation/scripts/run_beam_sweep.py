from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akkadian_mt.data import infer_columns
from akkadian_mt.metrics import compute_bleu, compute_chrfpp
from akkadian_mt.translator import MyTranslatorModel


def parse_beams(value: str) -> list[int]:
    beams = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not beams:
        raise argparse.ArgumentTypeError("At least one beam size is required")
    if any(beam < 1 for beam in beams):
        raise argparse.ArgumentTypeError("Beam sizes must be positive integers")
    return beams


def main() -> None:
    parser = argparse.ArgumentParser(description="Run beam-search ablation on a dev split.")
    parser.add_argument("--dataset", required=True, help="CSV with source and target columns")
    parser.add_argument("--model-name", default="google/byt5-small")
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--beams", type=parse_beams, default=[1, 4, 8])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--output", default="data/processed/beam_sweep.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if args.limit is not None:
        df = df.head(args.limit)
    columns = infer_columns(df, has_target=True)
    sources = df[columns.source].astype(str).tolist()
    references = df[columns.target].astype(str).tolist()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for beam in args.beams:
        translator = MyTranslatorModel(
            model_name=args.model_name,
            model_dir=args.model_dir,
            normalize=args.normalize,
            num_beams=beam,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        predictions = []
        for index, text in enumerate(sources, start=1):
            print(f"beams={beam} predicting {index}/{len(sources)}", flush=True)
            predictions.append(str(translator.predict(text, stream=False)))
        bleu = compute_bleu(predictions, references)
        chrfpp = compute_chrfpp(predictions, references)
        row = {"num_beams": beam, "bleu": bleu, "chrfpp": chrfpp}
        rows.append(row)
        print(f"beams={beam} BLEU={bleu:.4f} chrF++={chrfpp:.4f}")

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["num_beams", "bleu", "chrfpp"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
