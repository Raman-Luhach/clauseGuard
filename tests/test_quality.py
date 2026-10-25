"""Robustness suite and latency benchmark (on the synthetic ToS model)."""

from __future__ import annotations

import pytest

from clauseguard.evaluation.benchmark import run_benchmark
from clauseguard.evaluation.robustness import PERTURBATIONS, run_robustness
from clauseguard.ingestion.clean import clean_text
from clauseguard.paths import reports_dir

pytestmark = pytest.mark.slow
QUIET = lambda *_: None  # noqa: E731


def test_perturbations_survive_cleaning():
    import random

    text = "We may terminate your account at any time without notice for any reason."
    wrapped = PERTURBATIONS["pdf_line_wrap"](text, random.Random(0))
    assert "\n" in wrapped and clean_text(wrapped).text == text
    typo = PERTURBATIONS["typos"](text, random.Random(0))
    assert typo != text and len(typo) == len(text)
    assert clean_text(PERTURBATIONS["extra_whitespace"](text, random.Random(0))).text == text


def test_robustness_report(trained_tos):
    report = run_robustness("tos", n=20, log=QUIET)
    assert set(report["perturbations"]) == set(PERTURBATIONS)
    assert all(0 <= p["still_correct"] <= 1 for p in report["perturbations"].values())
    assert report["perturbations"]["all_caps"]["still_correct"] == 1.0  # model input is lowercased
    assert {"paraphrase", "neutral"} <= set(report["hand_written"])
    assert report["long_document"]["completed"] and report["long_document"]["words"] >= 50_000
    assert (reports_dir() / "tos_robustness.md").exists()


def test_benchmark_report(trained_tos):
    report = run_benchmark("tos", repeats=2, log=QUIET)
    assert set(report["sizes"]) == {"short", "medium", "long"}
    assert report["sizes"]["long"]["ms"]["total"]["p50"] > 0
    assert (reports_dir() / "tos_benchmark.md").exists()
