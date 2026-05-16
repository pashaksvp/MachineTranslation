from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pandas as pd

from akkadian_mt.data import infer_columns, load_parallel_csv
from akkadian_mt.logging_utils import get_logger
from akkadian_mt.normalization import normalize_akkadian


class MyTranslatorModel:
    """Train, load, and serve an Akkadian -> English translation model.

    This class is intentionally import-light: heavy ML libraries are imported only inside
    methods, so the API and tests stay usable before GPU dependencies are installed.
    """

    def __init__(
        self,
        model_name: str = "google/byt5-small",
        model_dir: str = "./model",
        max_source_length: int = 256,
        max_target_length: int = 256,
        normalize: bool = False,
        num_beams: int = 4,
        num_return_sequences: int = 1,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.normalize = normalize
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.logger = get_logger()
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        load_path = self.model_dir if self.model_dir.exists() else self.model_name
        self.logger.info("Loading model from %s", load_path)
        self._tokenizer = AutoTokenizer.from_pretrained(load_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self._model.eval()

    def train(self, dataset_path: str) -> None:
        """Fine-tune the forward model and save it to ./model/."""
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        self.logger.info("Starting training on dataset=%s", dataset_path)
        df = load_parallel_csv(dataset_path, normalize=self.normalize)
        dataset = Dataset.from_pandas(df, preserve_index=False)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            inputs = [f"translate Akkadian to English: {x}" for x in batch["source"]]
            model_inputs = tokenizer(
                inputs,
                max_length=self.max_source_length,
                truncation=True,
            )
            labels = tokenizer(
                text_target=batch["target"],
                max_length=self.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        args = Seq2SeqTrainingArguments(
            output_dir="outputs/byt5",
            learning_rate=5e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_steps=25,
            report_to=["wandb"],
            predict_with_generate=True,
            fp16=False,
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
            data_collator=collator,
        )
        trainer.train()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        self.logger.info("Training complete. Model saved to %s", self.model_dir)

    def predict(self, text: str, stream: bool = True) -> Iterator[str] | str:
        """Translate one string. When stream=True, yield text chunks."""
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        source = normalize_akkadian(text) if self.normalize else text
        prompt = f"translate Akkadian to English: {source}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_target_length,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            do_sample=False,
        )
        translation = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not stream:
            return translation

        def generate_chunks() -> Iterator[str]:
            for chunk in translation.split():
                yield chunk + " "
                time.sleep(0.02)

        return generate_chunks()

    def predict_file(self, dataset_path: str) -> None:
        """Load ./model/ and save Kaggle-format predictions to ./data/results.csv."""
        self.logger.info("Predicting file=%s", dataset_path)
        df = pd.read_csv(dataset_path)
        columns = infer_columns(df, has_target=False)
        predictions: list[str] = []
        for text in df[columns.source].astype(str).tolist():
            predictions.append(str(self.predict(text, stream=False)))

        output = pd.DataFrame({columns.id: df[columns.id], "translation": predictions})
        Path("data").mkdir(exist_ok=True)
        output.to_csv("data/results.csv", index=False)
        self.logger.info("Saved predictions to data/results.csv")
