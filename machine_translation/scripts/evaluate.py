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
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    columns = infer_columns(df, has_target=True)
    translator = MyTranslatorModel(model_dir=args.model_dir, normalize=args.normalize)
    predictions = [
        str(translator.predict(text, stream=False)) for text in df[columns.source].astype(str).tolist()
    ]
    references = df[columns.target].astype(str).tolist()
    print(f"BLEU: {compute_bleu(predictions, references):.4f}")
    print(f"chrF++: {compute_chrfpp(predictions, references):.4f}")


if __name__ == "__main__":
    main()
