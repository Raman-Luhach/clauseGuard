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


THRESHOLD_BETAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def any_risk(
    y: np.ndarray, p: np.ndarray, thresholds: list[float], risky: list[bool]
) -> tuple[float, float]:
    """(recall, precision) of "the unit has some risky label", over the ``risky`` labels."""
    idx = [j for j, r in enumerate(risky) if r]
    gold = y[:, idx].any(axis=1)
    pred = apply_thresholds(p[:, idx], [thresholds[j] for j in idx]).any(axis=1)
    hits = float((pred & gold).sum())
    recall = hits / gold.sum() if gold.any() else float("nan")
    precision = hits / pred.sum() if pred.any() else 0.0
    return float(recall), float(precision)


def tune_for_recall(
    y: np.ndarray,
    p: np.ndarray,
    risky: list[bool],
    target: float,
    min_precision: float | None = None,
) -> tuple[list[float], float, list[dict]]:
    """Per-label F-beta thresholds with the smallest beta whose any-risk recall reaches
    ``target``.

    A larger beta weighs recall more than precision. Betas are tried in increasing order; the
    search stops at the first one reaching the target, and never accepts a beta whose any-risk
    precision falls below ``min_precision`` (the previous beta is kept instead). Returns
    (thresholds, beta, one row per beta tried) so the choice can be reported.
    """
    tried, chosen = [], None
    for beta in THRESHOLD_BETAS:
        thresholds = [best_threshold(y[:, j], p[:, j], beta)[0] for j in range(y.shape[1])]
        recall, precision = any_risk(y, p, thresholds, risky)
        tried.append({"beta": beta, "any_recall": recall, "any_precision": precision})
        if min_precision is not None and precision < min_precision and chosen is not None:
            break
        chosen = (thresholds, beta)
        if recall >= target:
            break
    assert chosen is not None
    return chosen[0], chosen[1], tried
