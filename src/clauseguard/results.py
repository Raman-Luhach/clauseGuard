"""Move training results from a GPU machine to the Mac.

``pack_results`` (on the GPU machine) writes one archive with every finished run: metadata,
thresholds, metrics, validation predictions and error tables, plus the ONNX export of each
module's selected run (only the variant it serves). PyTorch weights stay behind.

``import_results`` (on the Mac) adds those runs to ``artifacts/`` next to the runs already
there, keeps the GPU machine's reports in a separate folder, and then, per module, serves
whichever is better on validation: the run served so far or the GPU session's selected run.
"""

from __future__ import annotations

import csv
import json
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from clauseguard.config import available_modules, load_module_config
from clauseguard.errors import ModelNotAvailableError
from clauseguard.models.runs import (
    LOG_COLUMNS,
    data_sha256,
    metric_value,
    read_json,
    resolve_run,
    select_run,
    selected_file,
)
from clauseguard.paths import artifacts_dir, experiments_log, reports_dir, root

SKIP_IN_RUN = {"model"}  # PyTorch weights: hundreds of MB per run, not needed for serving


def _copy_run(src: Path, dst: Path, *, with_onnx: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in SKIP_IN_RUN or (item.name == "onnx" and not with_onnx):
            continue
        if item.is_dir() and item.name == "onnx":
            meta = json.loads((item / "onnx.json").read_text())
            served = meta["files"][meta["serve"]]
            (dst / "onnx").mkdir(exist_ok=True)
            for name in ("onnx.json", "kind.json", "tokenizer.json", "export_report.json", served):
                if (item / name).exists():
                    shutil.copy2(item / name, dst / "onnx" / name)
            meta["files"] = {meta["serve"]: served}
            (dst / "onnx" / "onnx.json").write_text(json.dumps(meta, indent=2))
        elif item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)


def pack_results(out: Path, modules: list[str], log: Callable[[str], None] = print) -> Path:
    """Archive the runs of ``modules`` (and joint runs), reports, the experiment log and the
    data cards into ``out`` (a .tgz)."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "results"
        for module in [*modules, "joint"]:
            base = artifacts_dir(module)
            if not base.exists():
                continue
            selected = (
                read_json(selected_file(module))["run_id"]
                if selected_file(module).exists()
                else None
            )
            for run in sorted(base.iterdir()):
                if run.is_dir() and (run / "DONE").exists():
                    _copy_run(
                        run, stage / "runs" / module / run.name, with_onnx=run.name == selected
                    )
            if selected:
                shutil.copy2(selected_file(module), stage / "runs" / module / "selected.json")
        for name, src in (
            ("reports", reports_dir()),
            ("data_cards", root() / "data" / "data_cards"),
        ):
            if src.exists():
                shutil.copytree(src, stage / name)
        if experiments_log().exists():
            shutil.copy2(experiments_log(), stage / "log.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as tar:
            tar.add(stage, arcname="results")
    log(f"results -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def _merge_log(src: Path) -> int:
    """Append rows of ``src`` whose run_id is not yet in experiments/log.csv."""
    dest = experiments_log()
    seen = set()
    if dest.exists():
        with open(dest, encoding="utf-8") as f:
            seen = {row["run_id"] for row in csv.DictReader(f)}
    with open(src, encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["run_id"] not in seen]
    if not rows:
        return 0
    new = not dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=LOG_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        if new:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _safe_members(tar: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    members = []
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
            raise ValueError(f"unsafe path in archive: {m.name}")
        members.append(m)
    return members


def import_results(archive: Path, log: Callable[[str], None] = print) -> dict:
    """Add an archive's runs to artifacts/ and choose, per module, what to serve."""
    summary: dict = {"runs_added": 0, "modules": {}}
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, members=_safe_members(tar, Path(tmp)), filter="data")
        stage = Path(tmp) / "results"
        if not stage.exists():
            raise ValueError(f"{archive} is not a ClauseGuard results archive")
        lambda_selected: dict[str, str] = {}
        for module_dir in sorted((stage / "runs").iterdir()) if (stage / "runs").exists() else []:
            module = module_dir.name
            for run in sorted(p for p in module_dir.iterdir() if p.is_dir()):
                dst = artifacts_dir(module) / run.name
                if dst.exists():
                    continue
                shutil.copytree(run, dst)
                summary["runs_added"] += 1
            if (module_dir / "selected.json").exists():
                lambda_selected[module] = read_json(module_dir / "selected.json")["run_id"]
        if (stage / "reports").exists():
            target = reports_dir() / f"gpu-{stamp}"
            shutil.copytree(stage / "reports", target)
            summary["reports"] = str(target)
        if (stage / "log.csv").exists():
            summary["log_rows_added"] = _merge_log(stage / "log.csv")
    log(f"added {summary['runs_added']} runs; GPU reports -> {summary.get('reports')}")

    for module in available_modules():
        new = lambda_selected.get(module)
        if not new:
            continue
        metric = load_module_config(module).selection_metric
        new_score = metric_value(
            read_json(artifacts_dir(module) / new / "metrics_validation.json"), metric
        )
        try:
            current = resolve_run(module, None, metric).name
        except ModelNotAvailableError:
            current = None
        current_score = float("-inf")
        if current is not None:
            path = artifacts_dir(module) / current
            sidecar = path / "predictions_validation.json"
            scored_on = (
                read_json(sidecar)["data_sha256"]
                if sidecar.exists()
                else read_json(path / "meta.json")["data_sha256"]
            )
            if scored_on != data_sha256(module):
                # The unit table was rebuilt since this run was scored (e.g. better sentence
                # splitting): score it again on today's validation units, so both runs are
                # compared on the same data. Slow on a CPU for contracts (~30-40 min).
                log(f"{module}: re-scoring {current} on the rebuilt validation units ...")
                from clauseguard.evaluation.evaluate import evaluate_run

                evaluate_run(module, current, "validation")
            current_score = metric_value(
                read_json(artifacts_dir(module) / current / "metrics_validation.json"), metric
            )
        serve = new if new_score > current_score else current
        select_run(module, serve)
        summary["modules"][module] = {
            "previous": current,
            "previous_score": current_score,
            "gpu_selected": new,
            "gpu_score": new_score,
            "serving": serve,
            "metric": metric,
        }
        log(
            f"{module}: {metric} on validation: served {current} {current_score:.4f}, "
            f"GPU session {new} {new_score:.4f} -> serving {serve}"
        )
    return summary
