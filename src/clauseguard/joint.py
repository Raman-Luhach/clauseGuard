"""The joint model: one transformer for all three modules, trained on sentences (an experiment).

Every module already has a sentence-level table with namespaced labels: UNFAIR-ToS sentences,
OPP-115 sentences (labels from the annotated spans inside each segment) and CUAD sentences.
The joint model is one encoder with one output per label of all three modules (8 + 13 + 19).

A ToS sentence was never checked for contract or privacy labels, so its "no" for those labels
means "not annotated", not "absent". The loss therefore only counts the labels of a sentence's
own dataset (a loss mask), and each module is evaluated on its own labels and units, next to
its specialist model. For privacy, sentence scores are also pooled (max) per segment, so the
joint model can be compared with the paragraph-level specialist on the same segments.

It is not served: the report answers "does one shared model match three specialists?".
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clauseguard.config import ModelSpec, PosWeightSpec, load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.units import label_matrix, load_units
from clauseguard.evaluation.metrics import compute_metrics, macro_pr_auc
from clauseguard.models.runs import git_commit, read_json, resolve_run, write_json
from clauseguard.models.thresholds import tune_thresholds
from clauseguard.models.train import risky_mask, sample_negatives, subsample
from clauseguard.paths import processed_dir, reports_dir, root

MODULES = ("tos", "privacy", "contract")
SPEC = ModelSpec(
    type="transformer",
    checkpoint="nlpaueb/legal-bert-base-uncased",
    lr=3e-5,
    epochs=6,
    batch_size=32,
    patience=2,
)
MAX_LENGTH = 256
CONTRACT_NEGATIVE_RATIO = 4.0  # as for the contract specialist
POS_WEIGHT = PosWeightSpec(mode="sqrt", max=20.0)
SMOKE = {"n_train": 90, "n_eval": 45, "epochs": 1, "max_length": 32, "batch_size": 8}


def joint_dir(smoke: bool = False) -> Path:
    return root() / "artifacts" / ("smoke" if smoke else "runs") / "joint"


def joint_labels() -> list[str]:
    return [label for m in MODULES for label in load_module_config(m).label_ids]


def _source_table(module: str) -> pd.DataFrame:
    if module == "privacy":
        df = pd.read_parquet(processed_dir("privacy") / "sentences.parquet")
        df["labels"] = df["labels"].map(list)
        df["segment"] = df["unit_id"].str.rsplit(":", n=1).str[0]
    else:
        df = load_units(module)
        df["segment"] = df["unit_id"]
    return df.assign(source=module)


def joint_table(seed: int = 13, smoke: bool = False) -> pd.DataFrame:
    """Sentences of all three modules with their labels, split and source module."""
    parts = []
    for module in MODULES:
        df = _source_table(module)
        label_ids = load_module_config(module).label_ids
        train = df[(df["split"] == "train") & ~df["exclude_from_training"]]
        rest = df[df["split"] != "train"]
        if module == "contract" and not smoke:
            train = sample_negatives(train, label_ids, CONTRACT_NEGATIVE_RATIO, seed)
        if smoke:
            train = subsample(train, label_ids, SMOKE["n_train"] // 3, seed)
            rest = pd.concat(
                subsample(rest[rest["split"] == s], label_ids, SMOKE["n_eval"] // 3, seed)
                for s in ("validation", "test")
            )
        parts.append(pd.concat([train, rest]))
    cols = ["unit_id", "doc_id", "segment", "text", "labels", "split", "source"]
    return pd.concat(parts, ignore_index=True)[cols]


def loss_mask(sources: pd.Series, labels: list[str]) -> np.ndarray:
    """1 where the label belongs to the unit's own module."""
    prefix = np.array([label.split(":", 1)[0] for label in labels])
    return (sources.to_numpy()[:, None] == prefix[None, :]).astype(np.float32)


def _module_columns(labels: list[str], module: str) -> list[int]:
    return [j for j, label in enumerate(labels) if label.startswith(f"{module}:")]


def _per_module_score(df: pd.DataFrame, y: np.ndarray, labels: list[str]):
    """Early-stopping score: mean over modules of macro PR-AUC on the module's own units."""

    def score(p: np.ndarray) -> float:
        values = []
        for module in MODULES:
            rows = (df["source"] == module).to_numpy()
            cols = _module_columns(labels, module)
            if rows.any():
                values.append(macro_pr_auc(y[rows][:, cols], p[rows][:, cols]))
        return float(np.nanmean(values))

    return score


def _module_metrics(
    module: str, part: pd.DataFrame, y: np.ndarray, p: np.ndarray, labels, thresholds
) -> dict:
    cfg = load_module_config(module)
    rows = (part["source"] == module).to_numpy()
    cols = _module_columns(labels, module)
    yy, pp = y[rows][:, cols], p[rows][:, cols]
    th = [thresholds[labels[j]] for j in cols]
    out = {"sentences": compute_metrics(yy, pp, th, cfg.label_ids, risky=risky_mask(cfg))}
    if module == "privacy":  # pool sentences per segment: comparable with the specialist
        frame = pd.DataFrame(pp, columns=cfg.label_ids).assign(
            segment=part.loc[rows, "segment"].to_numpy()
        )
        pooled = frame.groupby("segment", sort=False).max()
        units = load_units("privacy").set_index("unit_id").loc[pooled.index]
        y_seg = label_matrix(units["labels"], cfg.label_ids)
        out["segments"] = compute_metrics(
            y_seg, pooled[cfg.label_ids].to_numpy(), th, cfg.label_ids, risky=risky_mask(cfg)
        )
    return out


def train_joint(
    *,
    seed: int = 13,
    smoke: bool = False,
    device: str = "auto",
    final: bool = False,
    checkpoint: str | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    from clauseguard.models.transformer import train_transformer

    labels = joint_labels()
    table = joint_table(seed, smoke)
    train = table[table["split"] == "train"].reset_index(drop=True)
    val = table[table["split"] == "validation"].reset_index(drop=True)
    y_train, y_val = label_matrix(train["labels"], labels), label_matrix(val["labels"], labels)
    log(
        "joint model: "
        + ", ".join(f"{m} {int((train['source'] == m).sum())}" for m in MODULES)
        + f" training sentences; {len(labels)} labels"
    )
    spec = SPEC.model_copy(update={"checkpoint": checkpoint}) if checkpoint else SPEC
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"joint-{spec.checkpoint.split('/')[-1]}-s{seed}-{stamp}"
    out = joint_dir(smoke) / run_id
    out.mkdir(parents=True)
    model, info = train_transformer(
        spec,
        labels,
        train["text"].tolist(),
        y_train,
        val["text"].tolist(),
        y_val,
        max_length=SMOKE["max_length"] if smoke else MAX_LENGTH,
        pos_weight=POS_WEIGHT,
        seed=seed,
        out_dir=out,
        device="cpu" if smoke else device,
        epochs=SMOKE["epochs"] if smoke else None,
        batch_size=SMOKE["batch_size"] if smoke else None,
        loss_mask=loss_mask(train["source"], labels),
        val_score=_per_module_score(val, y_val, labels),
        log=log,
    )
    p_val = model.predict_proba(val["text"].tolist())
    thresholds: dict[str, float] = {}
    for module in MODULES:
        rows = (val["source"] == module).to_numpy()
        cols = _module_columns(labels, module)
        objective = load_module_config(module).threshold_objective
        tuned = tune_thresholds(y_val[rows][:, cols], p_val[rows][:, cols], objective)
        thresholds.update({labels[j]: t for j, t in zip(cols, tuned, strict=True)})
    write_json(out / "thresholds.json", thresholds)
    metrics = {m: _module_metrics(m, val, y_val, p_val, labels, thresholds) for m in MODULES}
    write_json(out / "metrics_validation.json", metrics)
    if final:
        test = table[table["split"] == "test"].reset_index(drop=True)
        y_test = label_matrix(test["labels"], labels)
        p_test = model.predict_proba(test["text"].tolist())
        write_json(
            out / "metrics_test.json",
            {m: _module_metrics(m, test, y_test, p_test, labels, thresholds) for m in MODULES},
        )
    meta = {
        "run_id": run_id,
        "model": "joint",
        "checkpoint": spec.checkpoint,
        "seed": seed,
        "smoke": smoke,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "labels": labels,
        "n_train": {m: int((train["source"] == m).sum()) for m in MODULES},
        "spec": spec.model_dump(),
        "max_length": MAX_LENGTH,
        "train_info": info,
    }
    write_json(out / "meta.json", meta)
    (out / "DONE").touch()
    if smoke:
        shutil.rmtree(out / "model")
    for module in MODULES:
        m = metrics[module]["sentences"]
        log(
            f"joint / {module}: validation macro-F1 {m['macro_f1']:.3f}, any-risk recall "
            f"{m['any_risk']['recall']:.3f}"
        )
    if not smoke:
        write_joint_report(out)
    return out


def _specialist(module: str, split: str) -> tuple[str, dict] | None:
    try:
        path = resolve_run(module)
    except Exception:  # no specialist on this machine
        return None
    file = path / f"metrics_{split}.json"
    return (path.name, read_json(file)) if file.exists() else None


def write_joint_report(run: Path) -> Path:
    """Joint vs specialist, per module, on validation (and test when it was evaluated)."""
    meta = read_json(run / "meta.json")
    md = [
        "# Joint model vs one model per module",
        "",
        f"Joint run `{meta['run_id']}` ({meta['checkpoint']}, {len(meta['labels'])} labels, "
        f"training sentences: {meta['n_train']}). Each module is scored on its own labels and "
        "units; thresholds are tuned per label on validation, as for the specialists.",
        "",
    ]
    for split in ("validation", "test"):
        file = run / f"metrics_{split}.json"
        if not file.exists():
            continue
        metrics = read_json(file)
        rows = []
        for module in MODULES:
            unit = "segments" if module == "privacy" else "sentences"
            joint = metrics[module][unit]
            spec = _specialist(module, split)
            rows.append(
                [
                    module,
                    unit,
                    f"{joint['macro_f1']:.3f}",
                    f"{joint['any_risk']['recall']:.3f}",
                    f"{spec[1]['macro_f1']:.3f}" if spec else "–",
                    f"{spec[1]['any_risk']['recall']:.3f}" if spec else "–",
                    f"`{spec[0]}`" if spec else "–",
                ]
            )
        md += [f"## {split.capitalize()}", ""]
        md += md_table(
            [
                "module",
                "units",
                "joint macro F1",
                "joint any-risk recall",
                "specialist macro F1",
                "specialist any-risk recall",
                "specialist run",
            ],
            rows,
        )
        md.append("")
        sentence = metrics["privacy"]["sentences"]
        md += [
            f"Privacy at sentence level (joint only): macro F1 {sentence['macro_f1']:.3f}.",
            "",
        ]
    md += [
        "Specialist numbers come from each module's served run on this machine, with its own "
        "thresholds; the privacy specialist classifies whole segments.",
    ]
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "joint_results.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (run / "report.md").write_text(path.read_text(), encoding="utf-8")
    return path


def latest_joint_run() -> Path | None:
    runs = sorted(p for p in joint_dir().glob("joint-*") if (p / "DONE").exists())
    return runs[-1] if runs else None


def joint_summary(run: Path) -> dict:
    return json.loads((run / "metrics_validation.json").read_text())
