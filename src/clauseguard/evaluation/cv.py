"""Grouped k-fold cross-validation for the CPU baselines.

The privacy test set has only ~18 policies, so a single split gives noisy numbers. Here the
train + validation documents are split into k stratified folds; each fold is scored by a
model whose C and thresholds were chosen on an inner validation split of the other folds.
The official test split is never touched.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.splits import stratified_group_split
from clauseguard.data.units import label_matrix, load_units
from clauseguard.evaluation.metrics import compute_metrics
from clauseguard.models.baselines import TfidfModel
from clauseguard.models.runs import write_json
from clauseguard.models.thresholds import tune_thresholds
from clauseguard.models.train import risky_mask
from clauseguard.paths import reports_dir

SUMMARY = {
    "micro F1": ("micro", "f1"),
    "macro F1": ("macro_f1",),
    "macro PR-AUC": ("macro_pr_auc",),
    "any-risk precision": ("any_risk", "precision"),
    "any-risk recall": ("any_risk", "recall"),
}


def _get(metrics: dict, path: tuple[str, ...]) -> float:
    for key in path:
        metrics = metrics[key]
    return float(metrics)


def cross_validate(
    module: str,
    model_name: str = "tfidf",
    folds: int = 5,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> dict:
    cfg = load_module_config(module)
    spec = cfg.models[model_name]
    if spec.type != "tfidf":
        raise ValueError(
            "cross-validation is for the TF-IDF baseline; transformers use the fixed split"
        )
    label_ids = cfg.label_ids
    units = load_units(module)
    pool = units[units["split"] != "test"].reset_index(drop=True)
    label_sets: dict[str, set[str]] = defaultdict(set)
    for doc, labels in zip(pool["doc_id"], pool["labels"], strict=True):
        label_sets[doc].update(labels)
    fold_of = stratified_group_split(
        dict(label_sets), {f"fold{i}": 1.0 for i in range(folds)}, seed
    )

    results = []
    for i in range(folds):
        held_out = {d for d, f in fold_of.items() if f == f"fold{i}"}
        rest = {d: s for d, s in label_sets.items() if d not in held_out}
        inner = stratified_group_split(rest, {"train": 0.85, "validation": 0.15}, seed + i)
        role = pool["doc_id"].map(
            lambda d, inner=inner, held_out=held_out: "test" if d in held_out else inner[d]
        )
        train = pool[(role == "train") & ~pool["exclude_from_training"]]
        val, test = pool[role == "validation"], pool[role == "test"]
        y_train, y_val, y_test = (label_matrix(p["labels"], label_ids) for p in (train, val, test))
        model = TfidfModel(label_ids)
        info = model.fit(spec, train["text"].tolist(), y_train, val["text"].tolist(), y_val, seed)
        thresholds = tune_thresholds(
            y_val, model.predict_proba(val["text"].tolist()), cfg.threshold_objective
        )
        metrics = compute_metrics(
            y_test,
            model.predict_proba(test["text"].tolist()),
            thresholds,
            label_ids,
            risky=risky_mask(cfg),
        )
        results.append(
            {
                "fold": i,
                "documents": len(held_out),
                "units": len(test),
                "chosen_C": info["chosen_C"],
                "metrics": metrics,
            }
        )
        log(
            f"fold {i}: {len(held_out)} documents, macro-F1 {metrics['macro_f1']:.3f}, "
            f"micro-F1 {metrics['micro']['f1']:.3f}"
        )

    summary = {}
    for name, path in SUMMARY.items():
        values = [_get(r["metrics"], path) for r in results]
        summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    report = {
        "module": module,
        "model": model_name,
        "folds": folds,
        "seed": seed,
        "summary": summary,
        "folds_detail": results,
    }
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{module}_cv_{model_name}.json", report)
    md = [
        f"# {folds}-fold cross-validation: {module} / {model_name}",
        "",
        f"Train + validation documents ({len(label_sets)}) in {folds} stratified folds; the test "
        "split is untouched. Mean ± standard deviation over folds.",
        "",
    ]
    md += md_table(
        ["metric", "mean ± std"],
        [[k, f"{v['mean']:.3f} ± {v['std']:.3f}"] for k, v in summary.items()],
    )
    (out / f"{module}_cv_{model_name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
