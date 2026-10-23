"""Robustness suite: does the served model keep working when the input changes?

1. Perturbations of validation units the model already gets right (ALL CAPS, PDF-style line
   wrapping, clause numbering, extra whitespace, typos). Every perturbed text goes through the
   same cleaning as real input. Reported: share of units still fully correct.
2. Hand-written cases in configs/robustness/<module>.yaml: paraphrases and negations that
   must get specific labels, and neutral text that must not get any risky label.
3. Unrelated text (news-style paragraphs): risky flags per 1,000 words should be near zero.
4. A 50,000+ word document must be analysed completely; its run time is reported.
"""

from __future__ import annotations

import random
import re
import textwrap
import time
from collections.abc import Callable

import numpy as np
import yaml

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.units import label_matrix, load_units
from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.loader import load_text
from clauseguard.models.runs import write_json
from clauseguard.models.train import risky_mask
from clauseguard.modules.base import Analyzer
from clauseguard.paths import _CODE_ROOT, configs_dir, reports_dir

EXAMPLES = {
    "tos": "github-terms-of-service.txt",
    "privacy": "github-privacy-statement.txt",
    "contract": "cuad-test-contract.txt",
}
LONG_DOCUMENT_WORDS = 50_000


def _typos(text: str, rng: random.Random, n: int = 2) -> str:
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    for i in rng.sample(candidates, min(n, len(candidates))):
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


PERTURBATIONS: dict[str, Callable[[str, random.Random], str]] = {
    "all_caps": lambda t, rng: t.upper(),
    "pdf_line_wrap": lambda t, rng: "\n".join(textwrap.wrap(t, 60)),
    "clause_numbering": lambda t, rng: f"12.3(b) {t}",
    "extra_whitespace": lambda t, rng: re.sub(r" ", "   ", t) + "\n\n",
    "typos": _typos,
}


class _Predictor:
    """Applies the analyzer's own cleaning, text style, model and thresholds to raw texts."""

    def __init__(self, analyzer: Analyzer):
        self.analyzer = analyzer

    def __call__(self, texts: list[str]) -> np.ndarray:
        prepared = [self.analyzer.model_text(clean_text(t).text) for t in texts]
        return self.analyzer.model.predict_proba(prepared) >= self.analyzer.thresholds


def run_robustness(
    module: str,
    run_id: str | None = None,
    *,
    n: int = 200,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> dict:
    cfg = load_module_config(module)
    label_ids = cfg.label_ids
    analyzer = Analyzer(module, run_id)
    predict = _Predictor(analyzer)
    rng = random.Random(seed)
    report: dict = {"module": module, "run": analyzer.meta["run_id"]}

    # 1. perturbations of correctly predicted validation positives
    units = load_units(module)
    val = units[units["split"] == "validation"]
    y = label_matrix(val["labels"], label_ids).astype(bool)
    positive = y.any(axis=1)
    texts = val["text"].to_numpy()[positive]
    gold = y[positive]
    correct = (predict(list(texts)) == gold).all(axis=1)
    order = np.flatnonzero(correct)
    rng.shuffle(order)
    keep = order[:n]
    base_texts, base_gold = [texts[i] for i in keep], gold[keep]
    perturbations = {}
    for name, fn in PERTURBATIONS.items():
        perturbed = [fn(t, random.Random(seed + k)) for k, t in enumerate(base_texts)]
        still = (predict(perturbed) == base_gold).all(axis=1) if base_texts else np.array([])
        perturbations[name] = {
            "units": len(base_texts),
            "still_correct": float(still.mean()) if len(still) else float("nan"),
        }
        log(f"{name:18s} still correct: {perturbations[name]['still_correct']:.1%}")
    report["perturbations"] = perturbations

    # 2. hand-written cases
    cases = yaml.safe_load((configs_dir() / "robustness" / f"{module}.yaml").read_text())
    risky = [label for label, r in zip(label_ids, risky_mask(cfg), strict=True) if r]
    hand: dict[str, dict] = {}
    for kind in ("paraphrase", "negation"):
        items = cases.get(kind) or []
        if not items:
            continue
        preds = predict([c["text"] for c in items])
        results = []
        for case, row in zip(items, preds, strict=True):
            got = {label_ids[j] for j in np.flatnonzero(row)}
            results.append(
                {
                    "text": case["text"],
                    "expect": case["expect"],
                    "got": sorted(got),
                    "pass": set(case["expect"]) <= got,
                }
            )
        hand[kind] = {"pass_rate": float(np.mean([r["pass"] for r in results])), "cases": results}
        log(f"{kind:18s} pass rate: {hand[kind]['pass_rate']:.1%}")
    neutral = cases.get("neutral") or []
    if neutral:
        preds = predict(neutral)
        results = [
            {
                "text": t,
                "got": sorted(label_ids[j] for j in np.flatnonzero(row)),
                "pass": not any(row[label_ids.index(lab)] for lab in risky),
            }
            for t, row in zip(neutral, preds, strict=True)
        ]
        hand["neutral"] = {
            "pass_rate": float(np.mean([r["pass"] for r in results])),
            "cases": results,
        }
        log(f"{'neutral':18s} pass rate: {hand['neutral']['pass_rate']:.1%}")
    report["hand_written"] = hand

    # 3. unrelated text
    news = (configs_dir() / "robustness" / "out_of_scope.txt").read_text()
    doc = load_text(news)
    result = analyzer.analyze(doc)
    risky_flags = sum(1 for f in result.findings if f.severity in ("high", "medium", "low"))
    report["out_of_scope"] = {
        "words": doc.n_words,
        "risky_flags": risky_flags,
        "per_1000_words": round(1000 * risky_flags / doc.n_words, 2),
        "warned": any("No clauses typical" in w for w in result.warnings),
    }
    log(f"out of scope: {risky_flags} risky flags in {doc.n_words} words")

    # 4. very long document
    base = (_CODE_ROOT / "examples" / EXAMPLES[module]).read_text()
    repeats = -(-LONG_DOCUMENT_WORDS // len(base.split()))
    long_doc = load_text("\n\n".join([base] * repeats))
    started = time.perf_counter()
    long_result = analyzer.analyze(long_doc)
    seconds = time.perf_counter() - started
    report["long_document"] = {
        "words": long_doc.n_words,
        "units": long_result.n_units,
        "seconds": round(seconds, 2),
        "completed": True,
    }
    log(f"long document: {long_doc.n_words:,} words in {seconds:.1f} s")

    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{module}_robustness.json", report)
    md = [
        f"# Robustness: {module} ({report['run']})",
        "",
        "## Perturbations",
        "",
        f"Validation units the model already got right ({len(base_texts)}), perturbed and "
        "passed through the normal cleaning. Share still fully correct:",
        "",
    ]
    md += md_table(
        ["perturbation", "still correct"],
        [[k, f"{v['still_correct']:.1%}"] for k, v in perturbations.items()],
    )
    md += ["", "## Hand-written cases", ""]
    for kind, block in hand.items():
        md.append(f"**{kind}**: {block['pass_rate']:.0%} pass")
        md += [
            f"- {'✓' if c['pass'] else '✗'} {c['text']}  → got {c['got'] or 'nothing'}"
            for c in block["cases"]
        ]
        md.append("")
    oos, long_ = report["out_of_scope"], report["long_document"]
    md += [
        "## Unrelated text",
        "",
        f"{oos['risky_flags']} risky flags in {oos['words']} words of news-style text "
        f"({oos['per_1000_words']} per 1,000 words); low-coverage note shown: {oos['warned']}.",
        "",
        "## Very long document",
        "",
        f"{long_['words']:,} words ({long_['units']:,} units) analysed in {long_['seconds']} s.",
    ]
    (out / f"{module}_robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
