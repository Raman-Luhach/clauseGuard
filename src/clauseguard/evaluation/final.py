"""The final results table: every model on validation, and (with --final) on the test split.

Run after the GPU session. Validation numbers come from the stored runs. With ``final=True``
the test split is evaluated once for: the majority and TF-IDF baselines, and every seed of
the best transformer model, whose results are reported as mean ± standard deviation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.evaluation.evaluate import evaluate_run
from clauseguard.models.runs import list_runs, metric_value, read_json, selected_file, write_json
from clauseguard.paths import reports_dir

TEST_METRICS = {
    "macro F1": ("macro_f1",),
    "micro F1": ("micro", "f1"),
    "any-risk recall": ("any_risk", "recall"),
    "any-risk precision": ("any_risk", "precision"),
    "document macro F1": ("documents", "macro_f1"),
}


def _get(metrics: dict, path: tuple[str, ...]) -> float:
    try:
        return metric_value(metrics, ".".join(path))
    except (KeyError, TypeError):
        return float("nan")


def _fmt(values: list[float]) -> str:
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def final_report(
    modules: list[str], *, final: bool = False, log: Callable[[str], None] = print
) -> dict:
    report: dict = {}
    md = ["# ClauseGuard results", ""]
    for module in modules:
        cfg = load_module_config(module)
        runs = list_runs(module)
        if not runs:
            log(f"{module}: no runs")
            continue
        by_model: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            by_model[run["model"]].append(run)
        selected = (
            read_json(selected_file(module))["run_id"] if selected_file(module).exists() else None
        )
        validation = {
            model: {
                "seeds": len(rs),
                "macro_f1": [metric_value(r["metrics_validation"], "macro_f1") for r in rs],
            }
            for model, rs in by_model.items()
        }
        md += [
            f"## {cfg.display_name} (`{module}`)",
            "",
            f"Selected for serving: `{selected or 'best run by validation'}`.",
            "",
            "### Validation",
            "",
        ]
        md += md_table(
            ["model", "runs", "macro F1"],
            [[m, v["seeds"], _fmt(v["macro_f1"])] for m, v in validation.items()],
        )
        entry: dict = {"selected": selected, "validation": validation}

        if final:
            transformers = {
                m: rs for m, rs in by_model.items() if rs[0]["model_type"] == "transformer"
            }
            best_model = max(
                transformers, key=lambda m: np.mean(validation[m]["macro_f1"]), default=None
            )
            to_test = [m for m in by_model if by_model[m][0]["model_type"] != "transformer"]
            to_test += [best_model] if best_model else []
            test: dict[str, dict[str, list[float]]] = {}
            for model in to_test:
                for run in by_model[model]:
                    metrics, _ = evaluate_run(module, run["run_id"], "test", final=True)
                    log(f"{module} {run['run_id']}: test macro-F1 {metrics['macro_f1']:.3f}")
                    for name, path in TEST_METRICS.items():
                        test.setdefault(model, {}).setdefault(name, []).append(_get(metrics, path))
                    if "clauses" in metrics:
                        test[model].setdefault("clause recall", []).append(
                            metrics["clauses"]["micro_recall"]
                        )
                    if "lexglue" in metrics:
                        test[model].setdefault("LexGLUE micro / macro F1", []).append(
                            (metrics["lexglue"]["micro_f1"], metrics["lexglue"]["macro_f1"])
                        )
            names = list(TEST_METRICS) + (["clause recall"] if module == "contract" else [])
            md += ["", "### Test (evaluated once, at the end)", ""]
            md += md_table(
                ["model", *names],
                [[m, *(_fmt(t.get(n, [])) for n in names)] for m, t in test.items()],
            )
            for model, t in test.items():
                if "LexGLUE micro / macro F1" in t:
                    pairs = np.array(t["LexGLUE micro / macro F1"])
                    md.append(
                        f"\nLexGLUE-style (with a 'none' class) for {model}: micro "
                        f"{_fmt(list(pairs[:, 0]))}, macro {_fmt(list(pairs[:, 1]))}."
                    )
            entry["test"] = {
                m: {k: v for k, v in t.items() if k != "LexGLUE micro / macro F1"}
                for m, t in test.items()
            }
        md.append("")
        report[module] = entry

    for name in (
        "privacy_cv_tfidf",
        "tos_robustness",
        "privacy_robustness",
        "contract_robustness",
        "tos_benchmark",
        "privacy_benchmark",
        "contract_benchmark",
    ):
        if (reports_dir() / f"{name}.md").exists():
            md.append(f"- See also `{name}.md`.")
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "final_results.json", report)
    (out / "final_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
