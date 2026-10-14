"""Dataset loading and template-level splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "vendor_replies.csv"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]


def load(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"text", "label", "split", "template_id"} - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def make_splits(df: pd.DataFrame) -> Splits:
    """Train / validation / test, all split by template so no phrasing leaks across.

    The dataset already marks held-out test templates. From the remaining train
    templates, the highest-numbered template of each class becomes validation.
    """
    test = df[df["split"] == "test"]
    pool = df[df["split"] == "train"].copy()
    pool["t_num"] = pool["template_id"].str.rsplit("_", n=1).str[1].astype(int)
    val_ids = pool.groupby("label")["t_num"].transform("max")
    val = pool[pool["t_num"] == val_ids]
    train = pool[pool["t_num"] != val_ids]
    labels = sorted(df["label"].unique())
    drop = ["t_num"]
    return Splits(train.drop(columns=drop).reset_index(drop=True),
                  val.drop(columns=drop).reset_index(drop=True),
                  test.reset_index(drop=True), labels)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and average reply length, for the data analysis in the README."""
    stats = df.assign(words=df["text"].str.split().str.len())
    return (stats.groupby("label")
            .agg(samples=("text", "size"), templates=("template_id", "nunique"), avg_words=("words", "mean"))
            .round(1)
            .sort_values("samples", ascending=False))
