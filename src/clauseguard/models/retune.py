"""Re-tune the decision thresholds of a finished run, without training it again.

Thresholds come from the validation split only. A recall target (``target_any_recall``)
picks the smallest F-beta whose validation any-risk recall reaches it. The previous thresholds
are kept as ``thresholds_previous.json``; validation metrics, the risk-score calibration and
the monitoring baseline are recomputed, and so are the test metrics when the run already had
them (the change was decided on validation, so this is not tuning on the test set).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from clauseguard.config import load_module_config
from clauseguard.data.units import UNKNOWN_DOC, label_matrix, load_units
from clauseguard.evaluation.evaluate import evaluate_run, stored_predictions
from clauseguard.models.runs import read_json, resolve_run, write_json
from clauseguard.models.thresholds import tune_for_recall, tune_thresholds
from clauseguard.models.train import risky_mask
from clauseguard.monitoring import baseline_from_validation
from clauseguard.scoring.risk_score import calibrate_k, document_raw_scores


def validation_predictions(module: str, path: Path):
    """The validation units and the run's stored probabilities for them (computed if missing)."""
    cfg = load_module_config(module)
    units = load_units(module)
    val = units[units["split"] == "validation"].reset_index(drop=True)
    p = stored_predictions(path, "validation", val, cfg.label_ids)
    if p is None:  # missing, or made for an older version of the unit table
        evaluate_run(module, path.name, "validation")
        p = stored_predictions(path, "validation", val, cfg.label_ids)
    return val, p


def retune_run(
    module: str,
    run_id: str | None = None,
    *,
    target_any_recall: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    cfg = load_module_config(module)
    path = resolve_run(module, run_id, cfg.selection_metric)
    meta = read_json(path / "meta.json")
    if meta["model_type"] == "majority":
        raise ValueError("the majority baseline has no thresholds to tune")
    target = target_any_recall if target_any_recall is not None else cfg.target_any_recall
    val, p = validation_predictions(module, path)
    y = label_matrix(val["labels"], cfg.label_ids)
    before = read_json(path / "metrics_validation.json")

    if target is None:
        thresholds = tune_thresholds(y, p, cfg.threshold_objective)
        info: dict = {"method": cfg.threshold_objective}
    else:
        thresholds, beta, tried = tune_for_recall(
            y, p, risky_mask(cfg), target, cfg.min_any_precision
        )
        info = {"method": "target_any_recall", "target": target, "beta": beta, "tried": tried}
        log(f"{path.name}: beta {beta} (validation any-risk recall by beta: {tried})")

    previous = path / "thresholds_previous.json"
    if not previous.exists():
        shutil.copy2(path / "thresholds.json", previous)
    write_json(path / "thresholds.json", dict(zip(cfg.label_ids, thresholds, strict=True)))

    known = (val["doc_id"] != UNKNOWN_DOC).to_numpy()
    raw = document_raw_scores(val.loc[known, "doc_id"], p[known], thresholds, cfg)
    meta["risk_score_k"] = calibrate_k(raw.values(), cfg.risk_score_k)
    meta["thresholds"] = info
    meta["monitoring_baseline"] = baseline_from_validation(p, thresholds, cfg.label_ids)
    write_json(path / "meta.json", meta)

    after, _ = evaluate_run(module, path.name, "validation", use_stored=True)
    result = {"run": path.name, "thresholds": info, "validation_before": before}
    result["validation_after"] = after
    if (path / "metrics_test.json").exists():
        result["test_before"] = read_json(path / "metrics_test.json")
        result["test_after"], _ = evaluate_run(
            module, path.name, "test", final=True, use_stored=True
        )
    for split in ("validation", "test"):
        if f"{split}_after" not in result:
            continue
        b, a = result[f"{split}_before"], result[f"{split}_after"]
        log(
            f"{split}: macro-F1 {b['macro_f1']:.3f} -> {a['macro_f1']:.3f}  "
            f"any-risk recall {b['any_risk']['recall']:.3f} -> {a['any_risk']['recall']:.3f}  "
            f"precision {b['any_risk']['precision']:.3f} -> {a['any_risk']['precision']:.3f}"
        )
    return result
