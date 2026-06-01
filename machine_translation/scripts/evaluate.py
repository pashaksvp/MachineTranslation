from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akkadian_mt.data import infer_columns
from akkadian_mt.metrics import compute_bleu, compute_chrfpp
from akkadian_mt.translator import MyTranslatorModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV with source and target columns")
    parser.add_argument("--model-name", default="google/byt5-small")
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--predictions-out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    columns = infer_columns(df, has_target=True)
    translator = MyTranslatorModel(
        model_name=args.model_name,
        model_dir=args.model_dir,
        normalize=args.normalize,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    predictions = [
        str(translator.predict(text, stream=False)) for text in df[columns.source].astype(str).tolist()
    ]
    references = df[columns.target].astype(str).tolist()
    if args.predictions_out:
        output = pd.DataFrame(
            {
                "source": df[columns.source].astype(str),
                "reference": references,
                "prediction": predictions,
            }
        )
        Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.predictions_out, index=False)
    print(f"BLEU: {compute_bleu(predictions, references):.4f}")
    print(f"chrF++: {compute_chrfpp(predictions, references):.4f}")


if __name__ == "__main__":
    main()
