"""Per-label decision thresholds tuned on the validation set."""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.5


def best_threshold(y: np.ndarray, p: np.ndarray, beta: float = 1.0) -> tuple[float, float]:
    """Threshold maximising F-beta when predicting positive for ``p >= threshold``.

    Returns (threshold, best F-beta). The threshold sits halfway between the lowest accepted
    score and the next lower score, so small score shifts don't flip borderline cases. Labels
    without positives keep the default threshold.
    """
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0 or len(p) == 0:
        return DEFAULT_THRESHOLD, float("nan")
    order = np.argsort(-p, kind="stable")
    scores, hits = p[order], y[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    last_of_tie = np.r_[scores[1:] != scores[:-1], True]
    tp, fp, cut = tp[last_of_tie], fp[last_of_tie], scores[last_of_tie]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    b2 = beta**2
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    i = int(np.argmax(f))
    if f[i] <= 0:
        return DEFAULT_THRESHOLD, 0.0
    lower = cut[i + 1] if i + 1 < len(cut) else 0.0
    return float(max((cut[i] + lower) / 2, 1e-6)), float(f[i])


def tune_thresholds(y: np.ndarray, p: np.ndarray, objective: str = "f1") -> list[float]:
    beta = {"f1": 1.0, "f2": 2.0}[objective]
    return [best_threshold(y[:, j], p[:, j], beta)[0] for j in range(y.shape[1])]


def apply_thresholds(p: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return (p >= np.asarray(thresholds)[None, :]).astype(np.int8)
