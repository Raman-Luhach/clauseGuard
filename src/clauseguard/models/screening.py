"""Screening: the fast TF-IDF model decides which units the transformer needs to read.

Most sentences of a terms of service or a contract are harmless. The TF-IDF baseline scores
every unit in milliseconds; only units it gives some chance of a label go to the transformer,
the rest are treated as unflagged. The cut-off is chosen on validation as the highest one that
still passes ``flags_kept`` (default 99%) of the units the transformer flags on its own, so
the findings barely change while the transformer reads a fraction of the document.

``screen_run`` writes ``screening.json`` and a copy of the TF-IDF model (``screen/``) into the
transformer run, so the run stays self-contained when it is packaged for deployment.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clauseguard.models.base import load_model
from clauseguard.models.runs import data_sha256, list_runs, read_json, resolve_run, write_json

SCREEN_DIR = "screen"
SCREEN_FILE = "screening.json"
DEFAULT_FLAGS_KEPT = 0.99
MIN_KEEP_SHARE = 0.2  # always read at least this share of units with the transformer


def screen_scores(model, texts: list[str]) -> np.ndarray:
    """The screening score of each unit: its highest TF-IDF label probability."""
    if not texts:
        return np.zeros(0)
    return model.predict_proba(texts).max(axis=1)


def load_screen(run_path: Path):
    """(TF-IDF model, cut-off) when the run has screening, else None."""
    file = run_path / SCREEN_FILE
    if not file.exists() or not (run_path / SCREEN_DIR / "kind.json").exists():
        return None
    return load_model(run_path / SCREEN_DIR), float(read_json(file)["min_score"])


def _tfidf_run(module: str) -> Path:
    """The best TF-IDF run trained on the current unit table."""
    sha = data_sha256(module)
    runs = [r for r in list_runs(module) if r["model_type"] == "tfidf" and r["data_sha256"] == sha]
    if not runs:
        raise ValueError(f"no TF-IDF run on the current {module} data; run `make train-cpu`")
    return max(runs, key=lambda r: r["metrics_validation"]["macro_f1"])["path"]


def screen_run(
    module: str,
    run_id: str | None = None,
    *,
    flags_kept: float = DEFAULT_FLAGS_KEPT,
    log: Callable[[str], None] = print,
) -> dict:
    # Imported here: serving only needs load_screen / screen_scores above.
    from clauseguard.config import load_module_config
    from clauseguard.data.card import md_table
    from clauseguard.data.units import label_matrix, load_units
    from clauseguard.evaluation.evaluate import stored_predictions
    from clauseguard.evaluation.metrics import compute_metrics
    from clauseguard.models.train import risky_mask
    from clauseguard.paths import reports_dir

    cfg = load_module_config(module)
    path = resolve_run(module, run_id, cfg.selection_metric)
    if read_json(path / "meta.json")["model_type"] != "transformer":
        log(f"{module}: serves {path.name}, not a transformer; no screening needed")
        return {"run": path.name, "used": False, "table": []}
    units = load_units(module)
    val = units[units["split"] == "validation"].reset_index(drop=True)
    p = stored_predictions(path, "validation", val, cfg.label_ids)
    if p is None:
        from clauseguard.evaluation.evaluate import evaluate_run

        evaluate_run(module, path.name, "validation")
        p = stored_predictions(path, "validation", val, cfg.label_ids)
    y = label_matrix(val["labels"], cfg.label_ids)
    thresholds = np.array([read_json(path / "thresholds.json")[lab] for lab in cfg.label_ids])
    tfidf_path = _tfidf_run(module)
    scores = screen_scores(load_model(tfidf_path / "model"), val["text"].tolist())
    flagged = (p >= thresholds).any(axis=1)

    rows, chosen = [], None
    for keep_share in np.arange(1.0, MIN_KEEP_SHARE - 1e-9, -0.05):
        cut = float(np.quantile(scores, 1 - keep_share)) if keep_share < 1 else 0.0
        keep = scores >= cut
        kept = float(keep[flagged].mean()) if flagged.any() else 1.0
        m = compute_metrics(
            y,
            np.where(keep[:, None], p, 0.0),
            list(thresholds),
            cfg.label_ids,
            risky=risky_mask(cfg),
        )
        row = {
            "keep_share": round(float(keep.mean()), 3),
            "min_score": cut,
            "flags_kept": kept,
            "macro_f1": m["macro_f1"],
            "any_recall": m["any_risk"]["recall"],
            "any_precision": m["any_risk"]["precision"],
        }
        rows.append(row)
        if kept >= flags_kept:
            chosen = row
    if chosen is None or chosen["keep_share"] >= 0.95:
        log(f"{module}: screening would not save work at {flags_kept:.0%} of flags kept; not used")
        (path / SCREEN_FILE).unlink(missing_ok=True)
        return {"run": path.name, "used": False, "table": rows}

    if (path / SCREEN_DIR).exists():
        shutil.rmtree(path / SCREEN_DIR)
    shutil.copytree(tfidf_path / "model", path / SCREEN_DIR)
    full = rows[0]
    info = {
        **chosen,
        "tfidf_run": tfidf_path.name,
        "flags_kept_target": flags_kept,
        "without_screening": {k: full[k] for k in ("macro_f1", "any_recall", "any_precision")},
    }
    write_json(path / SCREEN_FILE, info)
    log(
        f"{module}: the transformer reads {chosen['keep_share']:.0%} of units (TF-IDF score >= "
        f"{chosen['min_score']:.4f}); flags kept {chosen['flags_kept']:.3f}; validation macro-F1 "
        f"{full['macro_f1']:.3f} -> {chosen['macro_f1']:.3f}, any-risk recall "
        f"{full['any_recall']:.3f} -> {chosen['any_recall']:.3f}"
    )
    md = [
        f"# Screening: {module} ({path.name})",
        "",
        f"TF-IDF run `{tfidf_path.name}` scores every unit; the transformer reads only units "
        f"scoring at least the cut-off. Chosen: the smallest share keeping >= {flags_kept:.0%} "
        "of the transformer's own flags (validation).",
        "",
    ]
    md += md_table(
        ["units read", "cut-off", "flags kept", "macro F1", "any-risk recall", "precision"],
        [
            [
                f"{r['keep_share']:.0%}" + (" ←" if r is chosen else ""),
                f"{r['min_score']:.4f}",
                f"{r['flags_kept']:.3f}",
                f"{r['macro_f1']:.3f}",
                f"{r['any_recall']:.3f}",
                f"{r['any_precision']:.3f}",
            ]
            for r in rows
        ],
    )
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{module}_screening.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"run": path.name, "used": True, **info, "table": rows}
