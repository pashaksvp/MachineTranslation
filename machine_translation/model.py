from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from akkadian_mt.translator import MyTranslatorModel


class My_Translator_Model:
    """Required homework API wrapper."""

    def __init__(self) -> None:
        self.model = MyTranslatorModel()

    def train(self, dataset_path: str) -> None:
        self.model.train(dataset_path)

    def predict(self, text: str, stream: bool = True) -> Iterator[str] | str:
        return self.model.predict(text, stream=stream)

    def predict_file(self, dataset_path: str) -> None:
        self.model.predict_file(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Akkadian -> English translator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--no-stream", action="store_true")

    predict_file_parser = subparsers.add_parser("predict-file")
    predict_file_parser.add_argument("--dataset", required=True)

    args = parser.parse_args()
    translator = My_Translator_Model()

    if args.command == "train":
        translator.train(args.dataset)
    elif args.command == "predict":
        result = translator.predict(args.text, stream=not args.no_stream)
        if isinstance(result, str):
            print(result)
        else:
            for token in result:
                print(token, end="", flush=True)
            print()
    elif args.command == "predict-file":
        translator.predict_file(args.dataset)


if __name__ == "__main__":
    main()
