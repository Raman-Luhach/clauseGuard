"""Train one model for one module and record the run."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clauseguard.config import ModuleConfig, load_module_config, load_taxonomy
from clauseguard.data.units import UNKNOWN_DOC, label_matrix, load_units
from clauseguard.evaluation.metrics import compute_metrics
from clauseguard.models.baselines import MajorityModel, TfidfModel
from clauseguard.models.runs import append_log, data_sha256, git_commit, new_run, write_json
from clauseguard.models.thresholds import DEFAULT_THRESHOLD, tune_thresholds
from clauseguard.monitoring import baseline_from_validation
from clauseguard.scoring.risk_score import calibrate_k, document_raw_scores


def risky_mask(cfg: ModuleConfig) -> list[bool]:
    taxonomy = load_taxonomy()
    return [taxonomy.labels[label].severity in ("high", "medium", "low") for label in cfg.label_ids]


def subsample(df: pd.DataFrame, label_ids: list[str], n: int, seed: int) -> pd.DataFrame:
    """Up to n units, half of them labelled when possible, so smoke runs see positives."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    labelled = np.flatnonzero(label_matrix(df["labels"], label_ids).any(axis=1))
    unlabelled = np.setdiff1d(np.arange(len(df)), labelled)
    n_pos = min(len(labelled), n // 2)
    picked = np.r_[
        rng.choice(labelled, n_pos, replace=False),
        rng.choice(unlabelled, min(len(unlabelled), n - n_pos), replace=False),
    ]
    return df.iloc[np.sort(picked)]


def sample_negatives(
    df: pd.DataFrame, label_ids: list[str], ratio: float, seed: int
) -> pd.DataFrame:
    """All labelled units plus ``ratio`` unlabelled units per labelled one (training only)."""
    y = label_matrix(df["labels"], label_ids).any(axis=1)
    labelled, unlabelled = np.flatnonzero(y), np.flatnonzero(~y)
    n_neg = min(len(unlabelled), int(round(ratio * len(labelled))))
    picked = np.r_[labelled, np.random.default_rng(seed).choice(unlabelled, n_neg, replace=False)]
    return df.iloc[np.sort(picked)]


def train_run(
    module: str,
    model_name: str,
    *,
    seed: int = 42,
    smoke: bool = False,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> Path:
    cfg = load_module_config(module)
    if model_name not in cfg.models:
        raise ValueError(
            f"unknown model {model_name!r} for {module}; choose from {', '.join(cfg.models)}"
        )
    spec = cfg.models[model_name]
    label_ids = cfg.label_ids
    units = load_units(module)
    train_df = units[(units["split"] == "train") & ~units["exclude_from_training"]]
    val_df = units[units["split"] == "validation"]
    excluded = int(((units["split"] == "train") & units["exclude_from_training"]).sum())
    if spec.negative_ratio is not None and not smoke:
        before = len(train_df)
        train_df = sample_negatives(train_df, label_ids, spec.negative_ratio, seed)
        log(f"negative sampling (ratio {spec.negative_ratio}): {before} -> {len(train_df)} units")
    if smoke:
        train_df = subsample(train_df, label_ids, cfg.smoke.n_train, seed)
        val_df = subsample(val_df, label_ids, cfg.smoke.n_eval, seed)
    y_train = label_matrix(train_df["labels"], label_ids)
    y_val = label_matrix(val_df["labels"], label_ids)
    train_texts, val_texts = train_df["text"].tolist(), val_df["text"].tolist()

    run_id, out = new_run(module, model_name, seed, smoke)
    log(f"run {run_id}: {len(train_df)} train / {len(val_df)} validation units -> {out}")
    info: dict = {}
    if spec.type == "majority":
        model = MajorityModel(label_ids).fit(y_train)
    elif spec.type == "tfidf":
        model = TfidfModel(label_ids)
        info = model.fit(spec, train_texts, y_train, val_texts, y_val, seed)
        log(
            f"tfidf: chose C={info['chosen_C']} "
            f"(validation macro PR-AUC by C: {info['val_macro_pr_auc_by_C']})"
        )
    else:
        from clauseguard.models.transformer import train_transformer

        max_length = cfg.smoke.max_length if smoke else (spec.max_length or cfg.max_length)
        model, info = train_transformer(
            spec,
            label_ids,
            train_texts,
            y_train,
            val_texts,
            y_val,
            max_length=max_length,
            pos_weight=cfg.pos_weight,
            seed=seed,
            out_dir=out,
            device="cpu" if smoke else device,
            epochs=cfg.smoke.epochs if smoke else None,
            batch_size=cfg.smoke.batch_size if smoke else None,
            log=log,
        )

    p_val = model.predict_proba(val_texts)
    if spec.type == "majority":
        thresholds = [DEFAULT_THRESHOLD] * len(label_ids)
    else:
        thresholds = tune_thresholds(y_val, p_val, cfg.threshold_objective)
    metrics = compute_metrics(
        y_val, p_val, thresholds, label_ids, risky=risky_mask(cfg), lexglue=module == "tos"
    )

    known = val_df["doc_id"] != UNKNOWN_DOC
    raw = document_raw_scores(val_df.loc[known, "doc_id"], p_val[known.to_numpy()], thresholds, cfg)
    k = calibrate_k(raw.values(), cfg.risk_score_k)

    if spec.type != "transformer":  # transformers save their best epoch while training
        model.save(out / "model")
    write_json(out / "thresholds.json", dict(zip(label_ids, thresholds, strict=True)))
    meta = {
        "run_id": run_id,
        "module": module,
        "model": model_name,
        "model_type": spec.type,
        "checkpoint": spec.checkpoint,
        "seed": seed,
        "smoke": smoke,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_sha256": data_sha256(module),
        "n_train": len(train_df),
        "n_train_excluded": excluded,
        "n_validation": len(val_df),
        "spec": spec.model_dump(),
        "train_info": info,
        "risk_score_k": k,
        "risk_calibration_docs": len(raw),
        "monitoring_baseline": baseline_from_validation(p_val, thresholds, label_ids),
    }
    write_json(out / "meta.json", meta)
    write_json(out / "metrics_validation.json", metrics)
    pd.DataFrame(p_val, columns=label_ids).assign(unit_id=val_df["unit_id"].to_numpy()).to_parquet(
        out / "predictions_validation.parquet", index=False
    )
    if not smoke:
        append_log(
            {
                **meta,
                "best_epoch": info.get("best_epoch", ""),
                "train_seconds": info.get("train_seconds", ""),
                "val_micro_f1": metrics["micro"]["f1"],
                "val_macro_f1": metrics["macro_f1"],
                "val_macro_pr_auc": metrics["macro_pr_auc"],
                "val_any_precision": metrics["any_risk"]["precision"],
                "val_any_recall": metrics["any_risk"]["recall"],
            }
        )
    (out / "DONE").touch()
    if smoke and spec.type == "transformer":
        # Saving and reloading the weights has been exercised; smoke runs are never served,
        # so don't keep hundreds of MB per check.
        shutil.rmtree(out / "model")
    log(
        f"validation: micro-F1 {metrics['micro']['f1']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
        f"macro PR-AUC {metrics['macro_pr_auc']:.3f}  "
        f"any-risk recall {metrics['any_risk']['recall']:.3f}  risk k={k}"
    )
    return out
