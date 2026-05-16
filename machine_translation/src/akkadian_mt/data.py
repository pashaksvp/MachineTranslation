from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from akkadian_mt.logging_utils import get_logger
from akkadian_mt.normalization import normalize_akkadian


@dataclass(frozen=True)
class ColumnSpec:
    source: str
    target: str | None = None
    id: str = "id"


def infer_columns(df: pd.DataFrame, has_target: bool = True) -> ColumnSpec:
    lower_to_original = {c.lower(): c for c in df.columns}
    source_candidates = ["akkadian", "source", "src", "transliteration", "text"]
    target_candidates = ["english", "target", "translation", "tgt"]

    source = next((lower_to_original[c] for c in source_candidates if c in lower_to_original), None)
    target = next((lower_to_original[c] for c in target_candidates if c in lower_to_original), None)
    id_col = lower_to_original.get("id", df.columns[0])

    if source is None:
        raise ValueError(f"Cannot infer source column from columns={list(df.columns)}")
    if has_target and target is None:
        raise ValueError(f"Cannot infer target column from columns={list(df.columns)}")
    return ColumnSpec(source=source, target=target, id=id_col)


def load_parallel_csv(path: str | Path, normalize: bool = False) -> pd.DataFrame:
    logger = get_logger()
    path = Path(path)
    logger.info("Loading parallel data from %s", path)
    df = pd.read_csv(path)
    columns = infer_columns(df, has_target=True)
    df = df[[columns.source, columns.target]].rename(
        columns={columns.source: "source", columns.target: "target"}
    )
    df = df.dropna(subset=["source", "target"]).drop_duplicates()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    if normalize:
        df["source"] = df["source"].map(normalize_akkadian)
    logger.info("Loaded %d clean parallel rows", len(df))
    return df


def make_dev_split(
    dataset_path: str | Path,
    output_dir: str | Path = "data/processed",
    dev_size: float = 0.1,
    seed: int = 42,
    normalize: bool = False,
) -> tuple[Path, Path]:
    df = load_parallel_csv(dataset_path, normalize=normalize)
    train_df, dev_df = train_test_split(df, test_size=dev_size, random_state=seed, shuffle=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_split.csv"
    dev_path = output_dir / "dev_split.csv"
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    get_logger().info("Saved train split=%s dev split=%s", train_path, dev_path)
    return train_path, dev_path
