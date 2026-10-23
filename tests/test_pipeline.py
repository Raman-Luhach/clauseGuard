from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from clauseguard.cli import app
from clauseguard.errors import InputError, ModelNotAvailableError
from clauseguard.evaluation.evaluate import evaluate_run
from clauseguard.models.runs import list_runs, resolve_run, select_run
from clauseguard.models.train import train_run
from clauseguard.pipeline import analyze, document_types
from tests.pdfgen import make_pdf

TOS_TEXT = """ACME TERMS OF SERVICE

Welcome to Acme. Please read these terms carefully before you start.
By using the service you agree to be bound by these terms.
We may terminate your account at any time without notice.
Any dispute will be resolved by binding individual arbitration.
These terms are governed by the laws of the State of California.
You can contact our support team by email if you have questions.
The service lets you share photos with your friends and family.
We hope you enjoy using our application every day of the week.
"""


def test_analyze_text_end_to_end(trained_tos):
    result = analyze("tos", text=TOS_TEXT)
    assert result.document_type == "tos" and result.model.run_id == trained_tos.name
    flagged = {f.labels[0].label for f in result.findings}
    assert {"tos:unilateral_termination", "tos:arbitration", "tos:contract_by_using"} <= flagged
    for finding in result.findings:
        assert result.text[finding.char_start : finding.char_end] == finding.text
        assert finding.explanation and finding.risk_group_name
        assert finding.page is None
    assert 0 <= result.risk_score <= 100 and result.risk_band in ("low", "medium", "high")
    assert sum(result.severity_counts.values()) == len(result.findings)
    assert "not legal advice" in result.disclaimer


def test_neutral_text_gets_a_low_coverage_note(trained_tos):
    neutral = " ".join(
        ["You can contact our support team by email."] * 3
        + ["The service lets you share photos with your friends."] * 3
        + ["Our office is open from Monday to Friday."] * 3
    )
    result = analyze("tos", text=neutral)
    assert not result.findings and result.risk_score == 0.0
    assert any("No clauses typical" in w for w in result.warnings)


def test_analyze_pdf_reports_pages(trained_tos):
    lines = [line for line in TOS_TEXT.splitlines() if line.strip()]
    pdf = make_pdf([lines[:5], lines[5:]])
    result = analyze("tos", pdf_bytes=pdf)
    pages = {f.labels[0].label: f.page for f in result.findings}
    assert pages["tos:unilateral_termination"] == 1
    assert pages["tos:choice_of_law"] == 2


def test_document_type_is_required(trained_tos):
    for bad in (None, "", "router", "TOS"):
        with pytest.raises(InputError) as err:
            analyze(bad, text=TOS_TEXT)
        assert err.value.code == "unknown_document_type"
    assert [t["id"] for t in document_types()] == ["tos", "privacy", "contract"]


def test_no_model_gives_clear_error(synthetic_tos):
    with pytest.raises(ModelNotAvailableError, match="clauseguard train tos"):
        analyze("tos", text=TOS_TEXT)


def test_runs_are_listed_selected_and_evaluated(synthetic_tos):
    tfidf = train_run("tos", "tfidf", log=lambda *_: None)
    majority = train_run("tos", "majority", log=lambda *_: None)
    assert {r["run_id"] for r in list_runs("tos")} == {tfidf.name, majority.name}
    assert resolve_run("tos") == tfidf  # majority is never served automatically
    select_run("tos", majority.name)
    assert resolve_run("tos") == majority
    with pytest.raises(ValueError, match="--final"):
        evaluate_run("tos", tfidf.name, "test")
    metrics, path = evaluate_run("tos", tfidf.name, "test", final=True)
    assert metrics["micro"]["f1"] > 0.9
    assert (path / "report_test.md").exists() and (path / "errors_test.csv").exists()
    meta = json.loads((tfidf / "meta.json").read_text())
    assert meta["risk_calibration_docs"] == 3 and meta["risk_score_k"] > 0


def test_smoke_run_is_kept_out_of_the_log(synthetic_tos, home):
    path = train_run("tos", "tfidf", smoke=True, log=lambda *_: None)
    assert "smoke" in path.parts
    assert not (home / "experiments" / "log.csv").exists()
    assert list_runs("tos") == []


# ---------------------------------------------------------------- CLI

runner = CliRunner()


def test_cli_analyze_requires_type(trained_tos, tmp_path):
    doc = tmp_path / "terms.txt"
    doc.write_text(TOS_TEXT)
    result = runner.invoke(app, ["analyze", str(doc)])
    assert result.exit_code != 0 and "--type" in result.output


def test_cli_analyze_json(trained_tos, tmp_path):
    doc = tmp_path / "terms.txt"
    doc.write_text(TOS_TEXT)
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["analyze", str(doc), "--type", "tos", "--json", "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["document_type"] == "tos" and report["findings"]


def test_cli_analyze_rejects_short_input(trained_tos, tmp_path):
    doc = tmp_path / "short.txt"
    doc.write_text("We may terminate your account.")
    result = runner.invoke(app, ["analyze", str(doc), "--type", "tos"])
    assert result.exit_code == 2 and "too_short" in result.output
