"""Training runs on disk, the experiment log, and choosing which run a module serves.

Layout: artifacts/runs/<module>/<run_id>/{model/, thresholds.json, meta.json,
metrics_<split>.json, predictions_<split>.parquet, DONE}. Smoke runs go to artifacts/smoke/
and are never logged or served.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from clauseguard.errors import ModelNotAvailableError
from clauseguard.paths import artifacts_dir, experiments_log, units_path

LOG_COLUMNS = [
    "run_id",
    "module",
    "model",
    "model_type",
    "checkpoint",
    "seed",
    "created_at",
    "git_commit",
    "data_sha256",
    "n_train",
    "n_validation",
    "best_epoch",
    "train_seconds",
    "val_micro_f1",
    "val_macro_f1",
    "val_macro_pr_auc",
    "val_any_precision",
    "val_any_recall",
]


def new_run(module: str, model_name: str, seed: int, smoke: bool) -> tuple[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{module}-{model_name}-s{seed}-{stamp}"
    path = artifacts_dir(module, smoke=smoke) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def data_sha256(module: str) -> str:
    return hashlib.sha256(units_path(module).read_bytes()).hexdigest()[:16]


def append_log(row: dict) -> None:
    path = experiments_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=LOG_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def list_runs(module: str) -> list[dict]:
    """Completed runs with their metadata and metrics, newest first."""
    runs = []
    base = artifacts_dir(module)
    if not base.exists():
        return runs
    for path in sorted(base.iterdir(), reverse=True):
        if not (path / "DONE").exists():
            continue
        meta = read_json(path / "meta.json")
        meta["path"] = path
        meta["metrics_validation"] = read_json(path / "metrics_validation.json")
        test = path / "metrics_test.json"
        meta["metrics_test"] = read_json(test) if test.exists() else None
        runs.append(meta)
    return runs


def metric_value(metrics: dict, name: str) -> float:
    value = metrics
    for key in name.split("."):
        value = value[key]
    return float(value)


def selected_file(module: str) -> Path:
    return artifacts_dir(module) / "selected.json"


def select_run(module: str, run_id: str) -> Path:
    path = artifacts_dir(module) / run_id
    if not (path / "DONE").exists():
        raise ModelNotAvailableError(f"run {run_id!r} not found or not finished")
    write_json(selected_file(module), {"run_id": run_id})
    return path


def resolve_run(module: str, run_id: str | None = None, selection_metric: str = "macro_f1") -> Path:
    """An explicit run, else the selected run, else the best completed run on validation."""
    if run_id:
        path = artifacts_dir(module) / run_id
        if not (path / "DONE").exists():
            raise ModelNotAvailableError(f"run {run_id!r} not found or not finished")
        return path
    selected = selected_file(module)
    if selected.exists():
        return resolve_run(module, read_json(selected)["run_id"])
    runs = [r for r in list_runs(module) if r["model_type"] != "majority"]
    if not runs:
        raise ModelNotAvailableError(
            f"no trained model for {module!r}; run `clauseguard train {module} --model tfidf`"
        )
    return max(runs, key=lambda r: metric_value(r["metrics_validation"], selection_metric))["path"]
