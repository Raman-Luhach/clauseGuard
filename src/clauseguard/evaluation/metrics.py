"""Multi-label metrics. Macro averages only include labels that occur in the evaluated data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from clauseguard.models.thresholds import apply_thresholds


def macro_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    present = [j for j in range(y.shape[1]) if y[:, j].any()]
    if not present:
        return float("nan")
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in present]))


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: list[float],
    label_ids: list[str],
    *,
    risky: list[bool] | None = None,
    lexglue: bool = False,
) -> dict:
    """Metrics for probabilities ``p`` against gold ``y`` (both shaped (n, labels)).

    ``risky`` marks the labels that count for the "any risky clause" metric (all by default).
    ``lexglue`` adds micro/macro F1 computed the LexGLUE way, with an extra "none" class.
    """
    pred = apply_thresholds(p, thresholds)
    present = [j for j in range(y.shape[1]) if y[:, j].any()]
    per_label = {}
    for j, label in enumerate(label_ids):
        support = int(y[:, j].sum())
        per_label[label] = {
            **_prf(y[:, j], pred[:, j]),
            "support": support,
            "predicted": int(pred[:, j].sum()),
            "pr_auc": float(average_precision_score(y[:, j], p[:, j])) if support else float("nan"),
            "threshold": float(thresholds[j]),
        }
    micro = _prf(y.ravel(), pred.ravel())
    risky_idx = [j for j, r in enumerate(risky or [True] * len(label_ids)) if r]
    any_true = y[:, risky_idx].any(axis=1)
    any_pred = pred[:, risky_idx].any(axis=1)
    any_score = p[:, risky_idx].max(axis=1) if risky_idx else np.zeros(len(y))
    out = {
        "n": int(len(y)),
        "micro": micro,
        "macro_f1": float(np.mean([per_label[label_ids[j]]["f1"] for j in present]))
        if present
        else float("nan"),
        "macro_pr_auc": macro_pr_auc(y, p),
        "macro_labels": len(present),
        "any_risk": {
            **_prf(any_true, any_pred),
            "pr_auc": float(average_precision_score(any_true, any_score))
            if any_true.any()
            else float("nan"),
        },
        "per_label": per_label,
    }
    if lexglue:
        y_ext = np.c_[y, ~y.any(axis=1)]
        pred_ext = np.c_[pred, ~pred.any(axis=1)]
        out["lexglue"] = {
            "micro_f1": float(f1_score(y_ext, pred_ext, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_ext, pred_ext, average="macro", zero_division=0)),
        }
    return out


def document_presence(
    doc_ids: Sequence[str], y: np.ndarray, pred: np.ndarray, label_ids: list[str]
) -> dict:
    """Per label, over documents: does the document contain the label anywhere?

    This is what the "missing protections" and "does this contract have X" features rely on.
    """
    frame = pd.DataFrame({"doc": list(doc_ids)})
    known = (frame["doc"] != "unknown").to_numpy()
    if not known.any():
        return {}
    docs_y = pd.DataFrame(y[known]).groupby(frame["doc"][known].to_numpy()).max().to_numpy()
    docs_p = pd.DataFrame(pred[known]).groupby(frame["doc"][known].to_numpy()).max().to_numpy()
    per_label = {}
    for j, label in enumerate(label_ids):
        per_label[label] = {
            **_prf(docs_y[:, j], docs_p[:, j]),
            "accuracy": float((docs_y[:, j] == docs_p[:, j]).mean()),
            "documents_with_label": int(docs_y[:, j].sum()),
        }
    present = [m for m in per_label.values() if m["documents_with_label"]]
    return {
        "n_documents": int(len(docs_y)),
        "macro_f1": float(np.mean([m["f1"] for m in present])) if present else float("nan"),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in per_label.values()])),
        "per_label": per_label,
    }


def clause_recall(
    spans: pd.DataFrame, units: pd.DataFrame, pred: np.ndarray, label_ids: list[str]
) -> dict:
    """Share of gold clause spans flagged with the right label in at least one overlapping unit.

    Independent of how spans were converted to unit labels. ``spans`` has doc_id, label, start,
    end; ``units`` has doc_id, char_start, char_end in the same coordinates and order as ``pred``.
    """
    index = {label: j for j, label in enumerate(label_ids)}
    by_doc: dict[str, list[tuple[int, int, int]]] = {}
    for row, (doc, start, end) in enumerate(
        zip(units["doc_id"], units["char_start"], units["char_end"], strict=True)
    ):
        by_doc.setdefault(doc, []).append((int(start), int(end), row))
    hits: dict[str, list[bool]] = {label: [] for label in label_ids}
    for doc, label, start, end in spans[["doc_id", "label", "start", "end"]].itertuples(
        index=False
    ):
        rows = [r for s, e, r in by_doc.get(doc, []) if s < end and e > start]
        hits[label].append(any(pred[r, index[label]] for r in rows))
    per_label = {
        label: {"recall": float(np.mean(h)) if h else float("nan"), "spans": len(h)}
        for label, h in hits.items()
    }
    all_hits = [x for h in hits.values() for x in h]
    present = [m["recall"] for m in per_label.values() if m["spans"]]
    return {
        "n_spans": len(all_hits),
        "micro_recall": float(np.mean(all_hits)) if all_hits else float("nan"),
        "macro_recall": float(np.mean(present)) if present else float("nan"),
        "per_label": per_label,
    }
