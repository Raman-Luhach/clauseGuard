"""The GPU training session: every transformer for every module, resumable.

For each module:
1. train each transformer in configs/<module>.yaml with the first seed;
2. the best one on validation (selection_metric) is the winner;
3. train the winner with the remaining seeds (results are reported as mean ± std);
4. select the best run on validation, unless the TF-IDF baseline beats it, for serving.
Finished runs are recognised (same model, seed, config and data) and skipped, so the session
can simply be started again after an interruption. Unfinished run folders are removed first.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from clauseguard.config import load_module_config
from clauseguard.models.runs import (
    data_sha256,
    list_runs,
    metric_value,
    read_json,
    select_run,
    write_json,
)
from clauseguard.models.train import train_run
from clauseguard.paths import artifacts_dir, reports_dir


def _matching(module: str, model: str, seed: int, smoke: bool) -> list[Path]:
    base = artifacts_dir(module, smoke=smoke)
    return sorted(base.glob(f"{module}-{model}-s{seed}-*")) if base.exists() else []


def find_completed(module: str, model: str, seed: int, smoke: bool) -> Path | None:
    spec = load_module_config(module).models[model].model_dump()
    sha = data_sha256(module)
    for path in reversed(_matching(module, model, seed, smoke)):
        if not (path / "DONE").exists():
            continue
        meta = read_json(path / "meta.json")
        if meta["spec"] == spec and meta["data_sha256"] == sha and meta["smoke"] == smoke:
            return path
    return None


def ensure_run(
    module: str, model: str, seed: int, *, smoke: bool, device: str, log: Callable[[str], None]
) -> Path:
    # Folders without DONE are left over from an interrupted run: they may hold large partial
    # checkpoints and are never used, so remove them first.
    for path in _matching(module, model, seed, smoke):
        if not (path / "DONE").exists():
            log(f"removing unfinished {path.name}")
            shutil.rmtree(path)
    done = find_completed(module, model, seed, smoke)
    if done is not None:
        log(f"skip {done.name}: already trained")
        return done
    return train_run(module, model, seed=seed, smoke=smoke, device=device, log=log)


def gpu_session(
    modules: list[str],
    *,
    smoke: bool = False,
    device: str = "auto",
    allow_cpu: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    if not smoke and not allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU found; start a GPU machine, or pass --allow-cpu")
    summary = {}
    for module in modules:
        cfg = load_module_config(module)
        metric = cfg.selection_metric
        transformers = [name for name, spec in cfg.models.items() if spec.type == "transformer"]
        if not transformers:
            raise ValueError(f"{module} has no transformer models configured")
        first, *others = cfg.seeds
        log(f"== {module}: comparing {', '.join(transformers)} (seed {first})")
        runs = {
            name: ensure_run(module, name, first, smoke=smoke, device=device, log=log)
            for name in transformers
        }
        scores = {
            name: metric_value(read_json(p / "metrics_validation.json"), metric)
            for name, p in runs.items()
        }
        winner = max(scores, key=scores.get)
        log(f"== {module}: winner {winner} ({metric} {scores[winner]:.4f}); seeds {others}")
        seed_runs = [runs[winner]] + [
            ensure_run(module, winner, seed, smoke=smoke, device=device, log=log) for seed in others
        ]
        seed_scores = {
            p.name: metric_value(read_json(p / "metrics_validation.json"), metric)
            for p in seed_runs
        }
        best = max(seed_runs, key=lambda p: seed_scores[p.name])
        selected = best.name
        if not smoke:
            baselines = [r for r in list_runs(module) if r["model_type"] == "tfidf"]
            top_baseline = max(
                baselines, key=lambda r: metric_value(r["metrics_validation"], metric), default=None
            )
            if (
                top_baseline
                and metric_value(top_baseline["metrics_validation"], metric) > seed_scores[selected]
            ):
                selected = top_baseline["run_id"]
                log(f"== {module}: the TF-IDF baseline is better on validation; serving it")
            select_run(module, selected)
        log(f"== {module}: selected {selected}")
        summary[module] = {
            "metric": metric,
            "comparison": scores,
            "winner": winner,
            "seed_runs": seed_scores,
            "selected": selected,
        }
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"gpu_session{'_smoke' if smoke else ''}.json", summary)
    return summary
