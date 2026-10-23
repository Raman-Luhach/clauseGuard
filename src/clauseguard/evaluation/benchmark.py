"""Latency of the full analysis (segmentation, classification, findings) on CPU.

Short, medium and long documents are cut from the module's example document; each is analysed
several times and the median and 95th percentile of every stage are reported. The result cache
is bypassed (the analyzer is called directly).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from clauseguard.data.card import md_table
from clauseguard.evaluation.robustness import EXAMPLES
from clauseguard.ingestion.loader import load_text
from clauseguard.models.runs import write_json
from clauseguard.modules.base import Analyzer
from clauseguard.paths import _CODE_ROOT, reports_dir

SIZES = {"short": 500, "medium": 2500, "long": 5000}


def run_benchmark(
    module: str,
    run_id: str | None = None,
    *,
    repeats: int = 5,
    log: Callable[[str], None] = print,
) -> dict:
    analyzer = Analyzer(module, run_id)
    words = (_CODE_ROOT / "examples" / EXAMPLES[module]).read_text().split()
    while len(words) < max(SIZES.values()):
        words = words + words
    analyzer.analyze(load_text(" ".join(words[:600])))  # warm-up: loads spaCy and the model
    report = {"module": module, "run": analyzer.meta["run_id"], "repeats": repeats, "sizes": {}}
    for name, n_words in SIZES.items():
        doc = load_text(" ".join(words[:n_words]))
        timings = [analyzer.analyze(doc).timings_ms for _ in range(repeats)]
        stages = {
            stage: {
                "p50": float(np.percentile([t[stage] for t in timings], 50)),
                "p95": float(np.percentile([t[stage] for t in timings], 95)),
            }
            for stage in timings[0]
        }
        report["sizes"][name] = {"words": doc.n_words, "ms": stages}
        log(
            f"{name:6s} {doc.n_words:5d} words: total p50 {stages['total']['p50']:.0f} ms, "
            f"p95 {stages['total']['p95']:.0f} ms"
        )
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{module}_benchmark.json", report)
    stage_names = list(report["sizes"]["short"]["ms"])
    md = [
        f"# Latency: {module} ({report['run']})",
        "",
        f"Milliseconds, median / 95th percentile over {repeats} runs, CPU, no cache.",
        "",
    ]
    md += md_table(
        ["document", "words", *stage_names],
        [
            [
                name,
                s["words"],
                *(f"{s['ms'][k]['p50']:.0f} / {s['ms'][k]['p95']:.0f}" for k in stage_names),
            ]
            for name, s in report["sizes"].items()
        ],
    )
    (out / f"{module}_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
