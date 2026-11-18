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
from clauseguard.models.base import load_model, serving_model_dir
from clauseguard.models.runs import data_sha256, read_json, resolve_run, write_json
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


def module_of(path: Path) -> str:
    return read_json(path / "meta.json")["module"]


def stored_predictions(path: Path, split: str, part: pd.DataFrame, label_ids: list[str]):
    """The run's saved probabilities for ``part`` (rows in its order), or None when they are
    missing or were made on another version of the unit table (unit ids are positions within
    a document, so after the data is rebuilt the same id can be a different sentence)."""
    stored = path / f"predictions_{split}.parquet"
    if not stored.exists():
        return None
    sidecar = path / f"predictions_{split}.json"
    made_on = (
        read_json(sidecar)["data_sha256"]
        if sidecar.exists()
        else read_json(path / "meta.json").get("data_sha256")
    )
    if made_on != data_sha256(module_of(path)):
        return None
    frame = pd.read_parquet(stored).set_index("unit_id")
    if not part["unit_id"].isin(frame.index).all() or not set(label_ids) <= set(frame.columns):
        return None
    return frame.loc[part["unit_id"], label_ids].to_numpy()


def evaluate_run(
    module: str,
    run_id: str | None = None,
    split: str = "validation",
    *,
    final: bool = False,
    device: str = "auto",
    use_stored: bool = False,
) -> tuple[dict, Path]:
    """Evaluate a run on a split. ``use_stored``: reuse ``predictions_<split>.parquet`` when
    the run has it (e.g. after changing only the thresholds) instead of running the model."""
    if split == "test" and not final:
        raise ValueError(
            "the test set is for final results only; pass --final once the model is chosen"
        )
    cfg = load_module_config(module)
    path = resolve_run(module, run_id, cfg.selection_metric)
    label_ids = cfg.label_ids
    thresholds_by_label = read_json(path / "thresholds.json")
    thresholds = [thresholds_by_label[label] for label in label_ids]
    units = load_units(module)
    part = units[units["split"] == split].reset_index(drop=True)
    y = label_matrix(part["labels"], label_ids)
    p = stored_predictions(path, split, part, label_ids) if use_stored else None
    if p is None:
        # PyTorch weights when the run still has them, else its ONNX export (runs copied
        # back from a GPU machine keep only ONNX).
        model_dir = path / "model" if (path / "model" / "kind.json").exists() else None
        model = load_model(model_dir or serving_model_dir(path), device=device)
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
    write_json(path / f"predictions_{split}.json", {"data_sha256": data_sha256(module)})
    errors.to_csv(path / f"errors_{split}.csv", index=False)
    confusions.to_csv(path / f"confusions_{split}.csv", index=False)
    (path / f"report_{split}.md").write_text(
        metrics_markdown(f"{path.name} on {split}", metrics, confusions), encoding="utf-8"
    )
    return metrics, path
