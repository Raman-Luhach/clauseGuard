"""Copy the served run of every module into a folder the deployed app can use.

The output mirrors ``artifacts/`` (``runs/<module>/<run_id>/`` plus ``selected.json``) so the
app finds it when ``CLAUSEGUARD_HOME`` points at the folder's parent. Transformer runs must
have an ONNX export; their PyTorch weights are left out, so the app runs without torch.
Predictions and error tables are left out too.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from clauseguard.config import load_module_config
from clauseguard.errors import ModelNotAvailableError
from clauseguard.models.runs import read_json, resolve_run

KEEP_FILES = (
    "DONE",
    "meta.json",
    "thresholds.json",
    "metrics_validation.json",
    "metrics_test.json",
)


def package_models(out: Path, modules: list[str], log: Callable[[str], None] = print) -> list[Path]:
    packaged = []
    for module in modules:
        run = resolve_run(module, None, load_module_config(module).selection_metric)
        model_type = read_json(run / "meta.json")["model_type"]
        has_onnx = (run / "onnx" / "onnx.json").exists()
        if model_type == "transformer" and not has_onnx:
            raise ModelNotAvailableError(
                f"{run.name} has no ONNX export; run `clauseguard export-onnx {module}` first"
            )
        target = out / "runs" / module / run.name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name in KEEP_FILES:
            if (run / name).exists():
                shutil.copy2(run / name, target / name)
        if has_onnx:
            onnx = json.loads((run / "onnx" / "onnx.json").read_text())
            (target / "onnx").mkdir()
            for name in ("onnx.json", "kind.json", "tokenizer.json", onnx["files"][onnx["serve"]]):
                shutil.copy2(run / "onnx" / name, target / "onnx" / name)
            onnx["files"] = {onnx["serve"]: onnx["files"][onnx["serve"]]}
            (target / "onnx" / "onnx.json").write_text(json.dumps(onnx, indent=2))
        else:
            shutil.copytree(run / "model", target / "model")
        (out / "runs" / module / "selected.json").write_text(json.dumps({"run_id": run.name}))
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
        log(f"{module}: {run.name} ({model_type}{', onnx' if has_onnx else ''}, {size:.0f} MB)")
        packaged.append(target)
    return packaged
