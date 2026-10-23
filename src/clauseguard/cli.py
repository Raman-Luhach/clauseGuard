"""`clauseguard` command line."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from clauseguard.config import DOCUMENT_TYPES, load_module_config
from clauseguard.errors import InputError, ModelNotAvailableError

app = typer.Typer(
    help="ClauseGuard: flag risky clauses in legal documents.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Download, prepare and describe datasets.", no_args_is_help=True)
models_app = typer.Typer(help="List and select trained runs.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
console = Console(stderr=True)

Module = Annotated[str, typer.Argument(help=f"Module: {', '.join(DOCUMENT_TYPES)}")]


def _preparers():
    from clauseguard.data import contract, privacy, tos

    return {
        "tos": (tos.prepare_tos, tos.card_extras),
        "privacy": (privacy.prepare_privacy, privacy.card_extras),
        "contract": (contract.prepare_contract, contract.card_extras),
    }


def _module_preparer(module: str):
    preparers = _preparers()
    if module not in preparers:
        raise typer.BadParameter(
            f"no data preparation for {module!r} yet; available: {', '.join(preparers)}"
        )
    return preparers[module]


@data_app.command("download")
def data_download(
    targets: Annotated[
        list[str], typer.Argument(help="all, tos, privacy, contract, or a dataset name")
    ],
    force: Annotated[bool, typer.Option(help="Download again even if present")] = False,
) -> None:
    """Download datasets and verify their SHA-256 checksums."""
    from clauseguard.data.download import download, load_manifest, resolve_targets

    manifest = load_manifest()
    for name in resolve_targets(targets):
        console.print(
            f"[bold]{name}[/]: {manifest[name]['description']} ({manifest[name]['license']})"
        )
        path = download(name, force=force)
        console.print(f"  ok -> {path}")


@data_app.command("prepare")
def data_prepare(module: Module) -> None:
    """Convert a module's dataset into its unit table (data/processed/<module>/units.parquet)."""
    from clauseguard.data.units import save_units
    from clauseguard.paths import units_path

    prepare, _ = _module_preparer(module)
    units = prepare()
    save_units(units, module)
    counts = units["split"].value_counts().to_dict()
    console.print(f"{len(units)} units {counts} -> {units_path(module)}")


@data_app.command("card")
def data_card(module: Module) -> None:
    """Write the data card for a module to data/data_cards/<module>.md."""
    from clauseguard.data.card import build_card
    from clauseguard.paths import data_cards_dir

    _, extras = _module_preparer(module)
    build_card(module, extras)
    console.print(f"written {data_cards_dir() / (module + '.md')}")


@app.command()
def train(
    module: Module,
    model: Annotated[str, typer.Option(help="Model name from configs/<module>.yaml")],
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    smoke: Annotated[bool, typer.Option(help="Tiny CPU run to check the pipeline works")] = False,
    device: Annotated[str, typer.Option(help="auto, cuda, mps or cpu")] = "auto",
) -> None:
    """Train one model and record the run (validation metrics, tuned thresholds)."""
    from clauseguard.models.train import train_run

    path = train_run(module, model, seed=seed, smoke=smoke, device=device, log=console.print)
    console.print(f"[green]done[/] -> {path}")


@app.command()
def smoke(modules: Annotated[list[str], typer.Argument(help="Modules to check")]) -> None:
    """Smoke-train every model in each module's config on CPU (run before the GPU session)."""
    from clauseguard.models.train import train_run

    failures = []
    for module in modules:
        for name in load_module_config(module).models:
            console.rule(f"{module} / {name}")
            try:
                train_run(module, name, smoke=True, log=console.print)
            except Exception as exc:  # report every failing model, not only the first
                failures.append(f"{module}/{name}: {exc}")
                console.print(f"[red]failed:[/] {exc}")
    if failures:
        raise RuntimeError("smoke runs failed:\n  " + "\n  ".join(failures))
    console.print("[green]all smoke runs passed[/]")


@app.command()
def cv(
    module: Module,
    model: Annotated[str, typer.Option(help="Baseline model name")] = "tfidf",
    folds: Annotated[int, typer.Option(help="Number of folds")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Grouped k-fold cross-validation of the TF-IDF baseline (never uses the test split)."""
    from clauseguard.evaluation.cv import cross_validate
    from clauseguard.paths import reports_dir

    report = cross_validate(module, model, folds, seed, log=console.print)
    for name, value in report["summary"].items():
        console.print(f"{name:20s} {value['mean']:.3f} ± {value['std']:.3f}")
    console.print(f"report -> {reports_dir() / f'{module}_cv_{model}.md'}")


@app.command("train-gpu")
def train_gpu(
    modules: Annotated[list[str], typer.Argument(help="Modules to train")],
    smoke: Annotated[bool, typer.Option(help="Tiny CPU run of the whole session")] = False,
    allow_cpu: Annotated[bool, typer.Option(help="Run without a CUDA GPU (slow)")] = False,
    device: Annotated[str, typer.Option(help="auto, cuda, mps or cpu")] = "auto",
) -> None:
    """The GPU session: every transformer, winner re-trained with 3 seeds, best one selected.
    Resumable: finished runs are skipped when you start it again."""
    from clauseguard.gpu import gpu_session

    summary = gpu_session(
        modules, smoke=smoke, device=device, allow_cpu=allow_cpu, log=console.print
    )
    for module, entry in summary.items():
        console.print(f"[green]{module}[/]: winner {entry['winner']}, selected {entry['selected']}")


@app.command("final-report")
def final_report_cmd(
    modules: Annotated[list[str], typer.Argument(help="Modules to report")],
    final: Annotated[
        bool, typer.Option(help="Also evaluate the test split (once, at the end)")
    ] = False,
) -> None:
    """Results table for the report: validation for every model; test with --final."""
    from clauseguard.evaluation.final import final_report
    from clauseguard.paths import reports_dir

    final_report(modules, final=final, log=console.print)
    console.print(f"report -> {reports_dir() / 'final_results.md'}")


@app.command("package-models")
def package_models_cmd(
    out: Annotated[Path, typer.Argument(help="Output folder, e.g. deploy/models")],
    modules: Annotated[list[str] | None, typer.Option("--module", help="Default: all")] = None,
) -> None:
    """Copy each module's served run (ONNX for transformers, no torch weights) for deployment."""
    from clauseguard.config import available_modules
    from clauseguard.deploy import package_models

    package_models(out, modules or available_modules(), log=console.print)
    console.print(f"packaged -> {out}")


@app.command()
def robustness(
    module: Module,
    run: Annotated[str | None, typer.Option(help="Run id (default: selected or best run)")] = None,
    n: Annotated[int, typer.Option(help="Validation units to perturb")] = 200,
) -> None:
    """Perturbations, paraphrases, negations, unrelated text and a 50k-word document."""
    from clauseguard.evaluation.robustness import run_robustness
    from clauseguard.paths import reports_dir

    run_robustness(module, run, n=n, log=console.print)
    console.print(f"report -> {reports_dir() / f'{module}_robustness.md'}")


@app.command()
def benchmark(
    module: Module,
    run: Annotated[str | None, typer.Option(help="Run id (default: selected or best run)")] = None,
    repeats: Annotated[int, typer.Option(help="Runs per document size")] = 5,
) -> None:
    """Latency per stage for short, medium and long documents (CPU, no cache)."""
    from clauseguard.evaluation.benchmark import run_benchmark
    from clauseguard.paths import reports_dir

    run_benchmark(module, run, repeats=repeats, log=console.print)
    console.print(f"report -> {reports_dir() / f'{module}_benchmark.md'}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    preload: Annotated[bool, typer.Option(help="Load every ready model at start-up")] = True,
) -> None:
    """Start the REST API and web UI."""
    import uvicorn

    from clauseguard.config import available_modules
    from clauseguard.pipeline import get_analyzer

    if preload:
        for module in available_modules():
            try:
                get_analyzer(module)
                console.print(f"loaded {module}")
            except ModelNotAvailableError as exc:
                console.print(f"[yellow]{module} unavailable:[/] {exc}")
    console.print(f"starting ClauseGuard on http://{host}:{port} ...")
    # uvicorn logs "Uvicorn running on ..." once the server actually accepts requests
    uvicorn.run("clauseguard.web.app:app", host=host, port=port, log_level="info")


@app.command()
def monitor(
    hours: Annotated[float | None, typer.Option(help="Only the last N hours")] = None,
) -> None:
    """Summarise logged traffic (metadata only) and flag drift against validation baselines."""
    from datetime import timedelta

    from clauseguard.config import available_modules
    from clauseguard.models.runs import read_json, resolve_run
    from clauseguard.monitoring import log_dir, summarize

    baselines = {}
    for module in available_modules():
        try:
            meta = read_json(resolve_run(module) / "meta.json")
        except Exception:  # no trained model for this module yet
            continue
        if "monitoring_baseline" in meta:
            baselines[module] = meta["monitoring_baseline"]
    summary = summarize(timedelta(hours=hours) if hours else None, baselines)
    if not summary:
        console.print(f"no events in {log_dir() / 'events.jsonl'}")
        return
    typer.echo(json.dumps(summary, indent=2))
    for doc_type, entry in summary.items():
        for d in entry["drift"]:
            console.print(f"[yellow]drift[/] {doc_type}: {d}")


@app.command("llm-compare")
def llm_compare_cmd(
    module: Module,
    n: Annotated[int, typer.Option(help="Units to sample (half labelled)")] = 200,
    split: Annotated[str, typer.Option(help="validation or test")] = "validation",
    final: Annotated[bool, typer.Option(help="Required for the test split")] = False,
    repeats: Annotated[int, typer.Option(help="Times the LLM labels the sample")] = 2,
) -> None:
    """Zero-shot Claude classification vs. the classifier (calls the API; costs money)."""
    from clauseguard.evaluation.llm_compare import llm_compare
    from clauseguard.paths import reports_dir

    llm_compare(module, n=n, split=split, final=final, repeats=repeats, log=console.print)
    console.print(f"report -> {reports_dir() / f'{module}_llm_compare.md'}")


@app.command("export-onnx")
def export_onnx_cmd(
    module: Module,
    run: Annotated[str | None, typer.Option(help="Run id (default: selected or best run)")] = None,
    quantize: Annotated[bool, typer.Option(help="Also write a dynamic INT8 model")] = True,
    compare: Annotated[bool, typer.Option(help="Compare F1/speed on validation and pick")] = True,
) -> None:
    """Export a fine-tuned transformer run to ONNX so it can be served without PyTorch."""
    from clauseguard.data.units import label_matrix, load_units
    from clauseguard.models.onnx import compare_backends, export_onnx
    from clauseguard.models.runs import read_json, resolve_run

    cfg = load_module_config(module)
    path = resolve_run(module, run, cfg.selection_metric)
    if read_json(path / "meta.json")["model_type"] != "transformer":
        raise ValueError(f"{path.name} is not a transformer run; only transformers are exported")
    export_onnx(path, quantize=quantize, log=console.print)
    if compare:
        units = load_units(module)
        val = units[units["split"] == "validation"]
        by_label = read_json(path / "thresholds.json")
        compare_backends(
            path,
            val["text"].tolist(),
            label_matrix(val["labels"], cfg.label_ids),
            [by_label[label] for label in cfg.label_ids],
            cfg.label_ids,
            log=console.print,
        )


@app.command()
def evaluate(
    module: Module,
    run: Annotated[str | None, typer.Option(help="Run id (default: selected or best run)")] = None,
    split: Annotated[str, typer.Option(help="validation or test")] = "validation",
    final: Annotated[bool, typer.Option(help="Required for the test split")] = False,
) -> None:
    """Evaluate a run; writes metrics, an error CSV and a markdown report into the run folder."""
    from clauseguard.evaluation.evaluate import evaluate_run

    try:
        metrics, path = evaluate_run(module, run, split, final=final)
    except (ValueError, ModelNotAvailableError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(
        f"{path.name} on {split}: micro-F1 {metrics['micro']['f1']:.3f}  "
        f"macro-F1 {metrics['macro_f1']:.3f}  macro PR-AUC {metrics['macro_pr_auc']:.3f}  "
        f"any-risk recall {metrics['any_risk']['recall']:.3f}"
    )
    console.print(f"report -> {path / f'report_{split}.md'}")


def _fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


@models_app.command("list")
def models_list(module: Module) -> None:
    """List completed runs with validation (and final test) metrics."""
    from clauseguard.models.runs import list_runs, read_json, selected_file

    selected_path = selected_file(module)
    selected = read_json(selected_path)["run_id"] if selected_path.exists() else None
    table = Table(title=f"{module} runs")
    for col in (
        "run",
        "model",
        "seed",
        "val micro-F1",
        "val macro-F1",
        "val PR-AUC",
        "test macro-F1",
    ):
        table.add_column(col)
    for run in list_runs(module):
        val, test = run["metrics_validation"], run["metrics_test"]
        name = run["run_id"] + (" [selected]" if run["run_id"] == selected else "")
        table.add_row(
            name,
            run["model"],
            str(run["seed"]),
            _fmt(val["micro"]["f1"]),
            _fmt(val["macro_f1"]),
            _fmt(val["macro_pr_auc"]),
            _fmt(test["macro_f1"]) if test else "–",
        )
    Console().print(table)


@models_app.command("select")
def models_select(module: Module, run: Annotated[str, typer.Argument(help="Run id")]) -> None:
    """Choose which run the analyzer, API and UI use for a module."""
    from clauseguard.models.runs import select_run

    path = select_run(module, run)
    console.print(f"{module} now uses {path.name}")


SEVERITY_STYLE = {
    "high": "bold red",
    "medium": "dark_orange",
    "low": "yellow",
    "protective": "green",
    "info": "cyan",
}


@app.command()
def analyze(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF or text file")],
    doc_type: Annotated[
        str, typer.Option("--type", "-t", help=f"Document type: {', '.join(DOCUMENT_TYPES)}")
    ],
    run: Annotated[str | None, typer.Option(help="Run id (default: selected or best run)")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the JSON report to this file")] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the JSON report to stdout")
    ] = False,
) -> None:
    """Analyse a document. You must say what kind of document it is with --type."""
    from clauseguard.pipeline import analyze as run_analysis

    try:
        result = run_analysis(doc_type, path=file, run_id=run)
    except InputError as exc:
        console.print(f"[red]{exc.message}[/] ({exc.code})")
        raise typer.Exit(2) from exc
    except ModelNotAvailableError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    report = result.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"report -> {out}")
    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    cfg = load_module_config(result.document_type)
    console.print(
        f"\n[bold]{cfg.display_name}[/] · {result.n_words:,} words · {result.n_units} units · "
        f"model {result.model.run_id}"
    )
    console.print(
        f"Risk score [bold]{result.risk_score}[/] / 100 ([bold]{result.risk_band}[/]) · "
        f"{len(result.findings)} flagged clauses {result.severity_counts}"
    )
    table = Table(show_lines=True)
    for col in ("severity", "category", "page", "clause", "explanation"):
        table.add_column(col, overflow="fold")
    for f in result.findings:
        style = SEVERITY_STYLE.get(f.severity, "")
        names = ", ".join(f"{label.name} ({label.score:.2f})" for label in f.labels)
        table.add_row(f"[{style}]{f.severity}[/]", names, str(f.page or "–"), f.text, f.explanation)
    Console().print(table)
    for warning in result.warnings:
        console.print(f"[yellow]note:[/] {warning}")
    console.print(f"[dim]{result.disclaimer}[/]")


def main() -> None:
    """Console entry point: expected failures print one clear line instead of a traceback."""
    try:
        app()
    except (InputError, ModelNotAvailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        if os.environ.get("CLAUSEGUARD_DEBUG"):
            raise
        console.print(f"[red]error:[/] {getattr(exc, 'message', exc)}")
        console.print("[dim]set CLAUSEGUARD_DEBUG=1 for the full traceback[/]")
        sys.exit(1)
