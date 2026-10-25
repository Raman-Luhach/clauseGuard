"""Export misclassified units for manual review and count which labels get confused."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from clauseguard.models.thresholds import apply_thresholds

NONE = "(none)"


def error_table(
    units: pd.DataFrame, y: np.ndarray, p: np.ndarray, thresholds: list[float], label_ids: list[str]
) -> pd.DataFrame:
    """One row per unit with at least one false positive or false negative.

    ``error_type`` is left empty for manual tagging (related labels confused, negation,
    segmentation error, label noise, ambiguous clause, ...).
    """
    pred = apply_thresholds(p, thresholds)
    rows = []
    for i in np.flatnonzero((pred != y).any(axis=1)):
        fp = [label_ids[j] for j in np.flatnonzero((pred[i] == 1) & (y[i] == 0))]
        fn = [label_ids[j] for j in np.flatnonzero((pred[i] == 0) & (y[i] == 1))]
        relevant = np.flatnonzero((pred[i] == 1) | (y[i] == 1))
        rows.append(
            {
                "unit_id": units["unit_id"].iat[i],
                "doc_id": units["doc_id"].iat[i],
                "text": units["text"].iat[i],
                "gold": "; ".join(label_ids[j] for j in np.flatnonzero(y[i])),
                "predicted": "; ".join(label_ids[j] for j in np.flatnonzero(pred[i])),
                "false_positives": "; ".join(fp),
                "false_negatives": "; ".join(fn),
                "scores": json.dumps({label_ids[j]: round(float(p[i, j]), 3) for j in relevant}),
                "error_type": "",
            }
        )
    columns = [
        "unit_id",
        "doc_id",
        "text",
        "gold",
        "predicted",
        "false_positives",
        "false_negatives",
        "scores",
        "error_type",
    ]
    return pd.DataFrame(rows, columns=columns)


def confusion_pairs(y: np.ndarray, pred: np.ndarray, label_ids: list[str]) -> pd.DataFrame:
    """Counts of (predicted, gold) label pairs among errors; "(none)" marks a pure false
    positive (predicted, gold none) or a pure miss (gold, predicted none)."""
    counts: Counter[tuple[str, str]] = Counter()
    for i in np.flatnonzero((pred != y).any(axis=1)):
        fp = [label_ids[j] for j in np.flatnonzero((pred[i] == 1) & (y[i] == 0))]
        fn = [label_ids[j] for j in np.flatnonzero((pred[i] == 0) & (y[i] == 1))]
        if fp and fn:
            counts.update((a, b) for a in fp for b in fn)
        else:
            counts.update((a, NONE) for a in fp)
            counts.update((NONE, b) for b in fn)
    rows = [{"predicted": a, "gold": b, "count": n} for (a, b), n in counts.most_common()]
    return pd.DataFrame(rows, columns=["predicted", "gold", "count"])
