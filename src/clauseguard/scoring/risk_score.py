"""Document risk score (0-100).

For each risky label found in a document we take its highest confidence p and its severity
weight w (high 3, medium 2, low 1). ``raw = sum(w * p)`` grows with the number and severity of
distinct risky categories, not with document length. ``score = 100 * (1 - exp(-raw / k))``
squashes it into 0-100; ``k`` is calibrated per run so that a typical validation document
scores 50 ("as risky as a typical document of this type").
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median

import numpy as np

from clauseguard.config import ModuleConfig, load_taxonomy
from clauseguard.scoring.rules import effective_severity, statements


def raw_risk(items: Iterable[tuple[str, str, float]], extra: float = 0.0) -> float:
    """``items`` are (label, severity, confidence) for every flag in a document."""
    weights = load_taxonomy().severity_weights
    best: dict[str, float] = {}
    for label, severity, score in items:
        value = weights.get(severity, 0.0) * score
        best[label] = max(best.get(label, 0.0), value)
    return sum(best.values()) + extra


def score_from_raw(raw: float, k: float) -> float:
    return round(100.0 * (1.0 - math.exp(-raw / k)), 1) if k > 0 else 0.0


def risk_band(score: float) -> str:
    bands = sorted(load_taxonomy().risk_bands.items(), key=lambda kv: kv[1])
    band = bands[0][0]
    for name, lower in bands:
        if score >= lower:
            band = name
    return band


def calibrate_k(raw_values: Iterable[float], default: float) -> float:
    """k such that the median document scores 50."""
    values = [v for v in raw_values if v > 0]
    if len(values) < 3:
        return default
    return round(median(values) / math.log(2), 3)


def document_raw_scores(
    doc_ids: Iterable[str], p: np.ndarray, thresholds: list[float], cfg: ModuleConfig
) -> dict[str, float]:
    """Raw risk per document from unit-level probabilities, scored like the analyzer does
    (severity modifiers and missing expected statements included)."""
    taxonomy = load_taxonomy()
    label_ids = cfg.label_ids
    items: dict[str, list[tuple[str, str, float]]] = {}
    present: dict[str, set[str]] = {}
    for doc, row in zip(doc_ids, p, strict=True):
        flagged = {label_ids[j] for j in np.flatnonzero(row >= np.asarray(thresholds))}
        present.setdefault(doc, set()).update(flagged)
        doc_items = items.setdefault(doc, [])
        for label in flagged:
            severity, _ = effective_severity(cfg, taxonomy, label, flagged)
            doc_items.append((label, severity, float(row[label_ids.index(label)])))
    return {
        doc: raw_risk(flags, extra=statements(cfg, taxonomy, present[doc])[2])
        for doc, flags in items.items()
    }
