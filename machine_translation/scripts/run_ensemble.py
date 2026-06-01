from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akkadian_mt.data import infer_columns
from akkadian_mt.metrics import compute_bleu, compute_chrfpp
from akkadian_mt.translator import MyTranslatorModel


def parse_model_dirs(value: str) -> list[str]:
    model_dirs = [item.strip() for item in value.split(",") if item.strip()]
    if len(model_dirs) < 2:
        raise argparse.ArgumentTypeError("Mini-ensemble requires at least two model dirs")
    return model_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints and use the best dev chrF++ model."
    )
    parser.add_argument("--dev-dataset", required=True, help="CSV with source and target columns")
    parser.add_argument("--test-dataset", default=None, help="Optional Kaggle test CSV")
    parser.add_argument("--model-dirs", type=parse_model_dirs, required=True)
    parser.add_argument("--model-name", default="google/byt5-small")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--metrics-out", default="data/processed/ensemble_metrics.csv")
    parser.add_argument("--submission-out", default="data/results.csv")
    args = parser.parse_args()

    dev_df = pd.read_csv(args.dev_dataset)
    dev_columns = infer_columns(dev_df, has_target=True)
    dev_sources = dev_df[dev_columns.source].astype(str).tolist()
    references = dev_df[dev_columns.target].astype(str).tolist()

    metric_rows: list[dict[str, str | float]] = []
    best_model_dir: str | None = None
    best_chrfpp = float("-inf")

    for model_dir in args.model_dirs:
        translator = MyTranslatorModel(
            model_name=args.model_name,
            model_dir=model_dir,
            normalize=args.normalize,
            num_beams=args.num_beams,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        predictions = [str(translator.predict(text, stream=False)) for text in dev_sources]
        bleu = compute_bleu(predictions, references)
        chrfpp = compute_chrfpp(predictions, references)
        metric_rows.append({"model_dir": model_dir, "bleu": bleu, "chrfpp": chrfpp})
        print(f"{model_dir}: BLEU={bleu:.4f} chrF++={chrfpp:.4f}")
        if chrfpp > best_chrfpp:
            best_chrfpp = chrfpp
            best_model_dir = model_dir

    assert best_model_dir is not None
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(metrics_out, index=False)
    print(f"best_model_dir={best_model_dir}")
    print(f"metrics saved: {metrics_out}")

    if args.test_dataset:
        translator = MyTranslatorModel(
            model_name=args.model_name,
            model_dir=best_model_dir,
            normalize=args.normalize,
            num_beams=args.num_beams,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        test_df = pd.read_csv(args.test_dataset)
        test_columns = infer_columns(test_df, has_target=False)
        predictions = [
            str(translator.predict(text, stream=False))
            for text in test_df[test_columns.source].astype(str).tolist()
        ]
        submission = pd.DataFrame({test_columns.id: test_df[test_columns.id], "translation": predictions})
        submission_out = Path(args.submission_out)
        submission_out.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_out, index=False)
        print(f"submission saved: {submission_out}")


if __name__ == "__main__":
    main()
