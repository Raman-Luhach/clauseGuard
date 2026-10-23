"""Evaluate a finished run on a split and write metrics, errors and a report into the run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from clauseguard.config import load_module_config
from clauseguard.data.units import UNKNOWN_DOC, label_matrix, load_units
from clauseguard.evaluation.error_analysis import confusion_pairs, error_table
from clauseguard.evaluation.metrics import clause_recall, compute_metrics, document_presence
from clauseguard.evaluation.report import metrics_markdown
from clauseguard.models.base import load_model
from clauseguard.models.runs import read_json, resolve_run, write_json
from clauseguard.models.thresholds import apply_thresholds
from clauseguard.models.train import risky_mask
from clauseguard.paths import processed_dir
from clauseguard.scoring.risk_score import document_raw_scores


def risk_rank_check(units, y, p, thresholds, cfg) -> dict:
    """Does the risk score order documents like the gold labels do? Spearman correlation between
    each document's raw risk from predictions and from the gold labels."""
    known = (units["doc_id"] != UNKNOWN_DOC).to_numpy()
    docs = units.loc[known, "doc_id"]
    predicted = document_raw_scores(docs, p[known], thresholds, cfg)
    gold = document_raw_scores(docs, y[known].astype(float), [0.5] * len(thresholds), cfg)
    names = sorted(gold)
    pred_values, gold_values = [predicted[d] for d in names], [gold[d] for d in names]
    if len(names) < 3 or len(set(pred_values)) < 2 or len(set(gold_values)) < 2:
        return {"documents": len(names), "spearman": float("nan")}  # not defined
    return {
        "documents": len(names),
        "spearman": float(spearmanr(pred_values, gold_values).statistic),
    }


def evaluate_run(
    module: str,
    run_id: str | None = None,
    split: str = "validation",
    *,
    final: bool = False,
    device: str = "auto",
) -> tuple[dict, Path]:
    if split == "test" and not final:
        raise ValueError(
            "the test set is for final results only; pass --final once the model is chosen"
        )
    cfg = load_module_config(module)
    path = resolve_run(module, run_id, cfg.selection_metric)
    label_ids = cfg.label_ids
    thresholds_by_label = read_json(path / "thresholds.json")
    thresholds = [thresholds_by_label[label] for label in label_ids]
    model = load_model(path / "model", device=device)

    units = load_units(module)
    part = units[units["split"] == split].reset_index(drop=True)
    y = label_matrix(part["labels"], label_ids)
    p = model.predict_proba(part["text"].tolist())
    metrics = compute_metrics(
        y, p, thresholds, label_ids, risky=risky_mask(cfg), lexglue=module == "tos"
    )

    pred = apply_thresholds(p, thresholds)
    metrics["documents"] = document_presence(part["doc_id"], y, pred, label_ids)
    metrics["risk_score"] = risk_rank_check(part, y, p, thresholds, cfg)
    spans_path = processed_dir(module) / "spans.parquet"
    if spans_path.exists():
        spans = pd.read_parquet(spans_path)
        metrics["clauses"] = clause_recall(spans[spans["split"] == split], part, pred, label_ids)

    errors = error_table(part, y, p, thresholds, label_ids)
    confusions = confusion_pairs(y, pred, label_ids)
    write_json(path / f"metrics_{split}.json", metrics)
    pd.DataFrame(p, columns=label_ids).assign(unit_id=part["unit_id"]).to_parquet(
        path / f"predictions_{split}.parquet", index=False
    )
    errors.to_csv(path / f"errors_{split}.csv", index=False)
    confusions.to_csv(path / f"confusions_{split}.csv", index=False)
    (path / f"report_{split}.md").write_text(
        metrics_markdown(f"{path.name} on {split}", metrics, confusions), encoding="utf-8"
    )
    return metrics, path
