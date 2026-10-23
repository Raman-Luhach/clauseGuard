"""The whole post-GPU workflow in miniature, with a tiny model instead of the real checkpoints:
GPU session -> resume -> final report -> ONNX export -> packaging -> serving without torch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from clauseguard.config import load_module_config, load_taxonomy
from clauseguard.data.units import save_units
from clauseguard.deploy import package_models
from clauseguard.evaluation.final import final_report
from clauseguard.gpu import gpu_session
from clauseguard.models.onnx import export_onnx
from clauseguard.models.runs import list_runs, read_json, select_run, selected_file
from clauseguard.models.train import train_run
from clauseguard.paths import _CODE_ROOT, artifacts_dir, reports_dir
from tests.conftest import synthetic_tos_units
from tests.test_pipeline import TOS_TEXT
from tests.test_transformer import tiny_checkpoint  # noqa: F401  (fixture)

pytestmark = pytest.mark.slow
QUIET = lambda *_: None  # noqa: E731


@pytest.fixture
def tiny_configs(tmp_path, monkeypatch, tiny_checkpoint):  # noqa: F811
    configs = tmp_path / "configs"
    shutil.copytree(_CODE_ROOT / "configs", configs)
    cfg = yaml.safe_load((configs / "tos.yaml").read_text())
    tiny = {
        "type": "transformer",
        "checkpoint": tiny_checkpoint,
        "epochs": 2,
        "batch_size": 16,
        "patience": 2,
    }
    cfg["models"] = {
        "majority": {"type": "majority"},
        "tfidf": {"type": "tfidf", "min_df": 1},
        "tiny-a": {**tiny, "lr": 1e-3},
        "tiny-b": {**tiny, "lr": 5e-4},
    }
    cfg["max_length"] = 32
    cfg["smoke"] = {"n_train": 60, "n_eval": 40, "epochs": 1, "max_length": 32, "batch_size": 8}
    (configs / "tos.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CLAUSEGUARD_CONFIGS", str(configs))
    load_module_config.cache_clear()
    load_taxonomy.cache_clear()
    yield configs
    load_module_config.cache_clear()
    load_taxonomy.cache_clear()


def test_smoke_session_does_not_select(home, tiny_configs):
    save_units(synthetic_tos_units(), "tos")
    summary = gpu_session(["tos"], smoke=True, log=QUIET)
    assert summary["tos"]["winner"] in ("tiny-a", "tiny-b")
    assert len(summary["tos"]["seed_runs"]) == 3
    assert not selected_file("tos").exists() and list_runs("tos") == []
    assert (reports_dir() / "gpu_session_smoke.json").exists()


def test_full_flow(home, tiny_configs, tmp_path):
    save_units(synthetic_tos_units(), "tos")
    train_run("tos", "tfidf", log=QUIET)
    train_run("tos", "majority", log=QUIET)
    summary = gpu_session(["tos"], allow_cpu=True, device="cpu", log=QUIET)["tos"]
    assert set(summary["comparison"]) == {"tiny-a", "tiny-b"}
    assert len(summary["seed_runs"]) == 3  # winner re-trained with the other seeds
    assert read_json(selected_file("tos"))["run_id"] == summary["selected"]

    # resuming trains nothing new, and an unfinished run folder is cleaned up and redone
    runs_before = sorted(p.name for p in artifacts_dir("tos").iterdir() if p.is_dir())
    broken = artifacts_dir("tos") / f"tos-{summary['winner']}-s87-20990101-000000"
    broken.mkdir()
    gpu_session(["tos"], allow_cpu=True, device="cpu", log=QUIET)
    assert not broken.exists()
    assert sorted(p.name for p in artifacts_dir("tos").iterdir() if p.is_dir()) == runs_before

    report = final_report(["tos"], final=True, log=QUIET)["tos"]
    assert set(report["test"]) == {"tfidf", "majority", summary["winner"]}
    assert len(report["test"][summary["winner"]]["macro F1"]) == 3

    # serve a transformer: export to ONNX, package, analyse in a process without torch
    best = max(summary["seed_runs"], key=summary["seed_runs"].get)
    select_run("tos", best)
    export_onnx(artifacts_dir("tos") / best, log=QUIET)
    deploy_home = tmp_path / "deploy"
    package_models(deploy_home / "artifacts", ["tos"], log=QUIET)
    packaged = deploy_home / "artifacts" / "runs" / "tos" / best
    assert (packaged / "onnx" / "onnx.json").exists() and not (packaged / "model").exists()

    script = (
        "import sys, json\n"
        "sys.modules['torch'] = None\nsys.modules['transformers'] = None\n"
        "from clauseguard.pipeline import analyze\n"
        "r = analyze('tos', text=sys.stdin.read())\n"
        "print(json.dumps({'run': r.model.run_id, 'units': r.n_units}))\n"
    )
    env = {
        **os.environ,
        "CLAUSEGUARD_HOME": str(deploy_home),
        "CLAUSEGUARD_CONFIGS": str(tiny_configs),
    }
    done = subprocess.run(
        [sys.executable, "-c", script],
        input=TOS_TEXT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1])["run"] == best
