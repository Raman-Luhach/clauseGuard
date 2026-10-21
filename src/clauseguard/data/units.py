"""The unit table: one row per text unit (sentence or segment) a module trains on.

Columns: doc_id, unit_id, unit_type, text, char_start, char_end, labels, split,
exclude_from_training, meta.
``labels`` holds namespaced label ids; ``exclude_from_training`` marks units whose labels are
uncertain (they still count in evaluation); ``meta`` is a JSON string with module details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clauseguard.paths import units_path

COLUMNS = [
    "doc_id",
    "unit_id",
    "unit_type",
    "text",
    "char_start",
    "char_end",
    "labels",
    "split",
    "exclude_from_training",
    "meta",
]
SPLITS = ("train", "validation", "test")
UNKNOWN_DOC = "unknown"


def validate_units(df: pd.DataFrame, label_ids: list[str]) -> None:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"unit table is missing columns: {sorted(missing)}")
    if not df["unit_id"].is_unique:
        raise ValueError("unit_id values must be unique")
    bad_splits = set(df["split"]) - set(SPLITS)
    if bad_splits:
        raise ValueError(f"unknown splits: {sorted(bad_splits)}")
    if (df["text"].str.strip() == "").any():
        raise ValueError("unit table contains empty texts")
    unknown = {label for labels in df["labels"] for label in labels} - set(label_ids)
    if unknown:
        raise ValueError(f"labels not in the module config: {sorted(unknown)}")
    known_docs = df[df["doc_id"] != UNKNOWN_DOC]
    leaking = known_docs.groupby("doc_id")["split"].nunique()
    if (leaking > 1).any():
        raise ValueError(
            f"documents in more than one split: {list(leaking[leaking > 1].index)[:5]}"
        )


def save_units(df: pd.DataFrame, module: str) -> None:
    path = units_path(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(path, index=False)


def load_units(module: str) -> pd.DataFrame:
    path = units_path(module)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `clauseguard data download {module}` and "
            f"`clauseguard data prepare {module}` first"
        )
    df = pd.read_parquet(path)
    df["labels"] = df["labels"].map(list)
    if "exclude_from_training" not in df:  # tables built before the column existed
        df["exclude_from_training"] = False
    return df


def label_matrix(labels: pd.Series, label_ids: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(label_ids)}
    y = np.zeros((len(labels), len(label_ids)), dtype=np.int8)
    for row, unit_labels in enumerate(labels):
        for label in unit_labels:
            y[row, index[label]] = 1
    return y
