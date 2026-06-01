from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from akkadian_mt.translator import MyTranslatorModel


class My_Translator_Model:
    """Required homework API wrapper."""

    def __init__(
        self,
        model_name: str = "google/byt5-small",
        model_dir: str = "./model",
        max_source_length: int = 256,
        max_target_length: int = 256,
        normalize: bool = False,
        num_beams: int = 4,
        learning_rate: float = 5e-4,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.model = MyTranslatorModel(
            model_name=model_name,
            model_dir=model_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            normalize=normalize,
            num_beams=num_beams,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            seed=seed,
        )

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
    train_parser.add_argument("--model-name", default="google/byt5-small")
    train_parser.add_argument("--model-dir", default="./model")
    train_parser.add_argument("--max-source-length", type=int, default=256)
    train_parser.add_argument("--max-target-length", type=int, default=256)
    train_parser.add_argument("--normalize", action="store_true")
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_parser.add_argument("--num-train-epochs", type=float, default=3.0)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--model-name", default="google/byt5-small")
    predict_parser.add_argument("--model-dir", default="./model")
    predict_parser.add_argument("--max-source-length", type=int, default=256)
    predict_parser.add_argument("--max-target-length", type=int, default=256)
    predict_parser.add_argument("--normalize", action="store_true")
    predict_parser.add_argument("--num-beams", type=int, default=4)
    predict_parser.add_argument("--no-stream", action="store_true")

    predict_file_parser = subparsers.add_parser("predict-file")
    predict_file_parser.add_argument("--dataset", required=True)
    predict_file_parser.add_argument("--model-name", default="google/byt5-small")
    predict_file_parser.add_argument("--model-dir", default="./model")
    predict_file_parser.add_argument("--max-source-length", type=int, default=256)
    predict_file_parser.add_argument("--max-target-length", type=int, default=256)
    predict_file_parser.add_argument("--normalize", action="store_true")
    predict_file_parser.add_argument("--num-beams", type=int, default=4)

    args = parser.parse_args()
    translator = My_Translator_Model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        normalize=args.normalize,
        num_beams=getattr(args, "num_beams", 4),
        learning_rate=getattr(args, "learning_rate", 5e-4),
        per_device_train_batch_size=getattr(args, "per_device_train_batch_size", 4),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 4),
        num_train_epochs=getattr(args, "num_train_epochs", 3.0),
        seed=getattr(args, "seed", 42),
    )

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
