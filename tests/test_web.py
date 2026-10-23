"""REST API and web UI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clauseguard.monitoring import log_dir, read_events, summarize
from clauseguard.web.app import create_app
from tests.pdfgen import make_pdf
from tests.test_pipeline import TOS_TEXT


@pytest.fixture
def client(home):
    return TestClient(create_app())


def test_health_without_models_is_503(client):
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["status"] == "unavailable"
    types = client.get("/document-types").json()
    assert [t["id"] for t in types] == ["tos", "privacy", "contract"]
    assert not any(t["ready"] for t in types)


def test_analyze_text_and_pdf(trained_tos, client):
    health = client.get("/health").json()
    assert health["modules"]["tos"]["ready"] and not health["modules"]["privacy"]["ready"]
    response = client.post("/analyze", data={"document_type": "tos", "text": TOS_TEXT})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["findings"] and body["disclaimer"]
    lines = [line for line in TOS_TEXT.splitlines() if line.strip()]
    pdf = make_pdf([lines[:5], lines[5:]])
    response = client.post(
        "/analyze",
        data={"document_type": "tos"},
        files={"file": ("terms.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200 and response.json()["n_pages"] == 2
    response = client.post(
        "/analyze",
        data={"document_type": "tos"},
        files={"file": ("terms.txt", TOS_TEXT.encode(), "text/plain")},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("data", "files", "status", "code"),
    [
        ({"text": TOS_TEXT}, None, 422, "unknown_document_type"),
        ({"document_type": "router", "text": TOS_TEXT}, None, 422, "unknown_document_type"),
        ({"document_type": "tos"}, None, 422, "missing_input"),
        ({"document_type": "tos", "text": "Too short."}, None, 422, "too_short"),
        (
            {"document_type": "tos"},
            {"file": ("scan.pdf", make_pdf([[], []]), "application/pdf")},
            422,
            "scanned_pdf",
        ),
        ({"document_type": "privacy", "text": TOS_TEXT}, None, 503, "model_unavailable"),
    ],
)
def test_analyze_errors(trained_tos, client, data, files, status, code):
    response = client.post("/analyze", data=data, files=files)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_upload_size_limit(trained_tos, client, monkeypatch):
    import clauseguard.web.app as web

    monkeypatch.setattr(web, "MAX_PDF_BYTES", 1000)
    response = client.post(
        "/analyze",
        data={"document_type": "tos"},
        files={"file": ("big.txt", b"x" * 2000, "text/plain")},
    )
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"


def test_monitoring_logs_metadata_only(trained_tos, client):
    client.post("/analyze", data={"document_type": "tos", "text": TOS_TEXT})
    client.post("/analyze", data={"document_type": "tos", "text": "Too short."})
    client.post(
        "/feedback",
        json={
            "document_type": "tos",
            "model_run": trained_tos.name,
            "label": "tos:arbitration",
            "agree": False,
        },
    )
    raw = (log_dir() / "events.jsonl").read_text()
    assert "binding individual arbitration" not in raw and "Too short" not in raw
    events = read_events()
    assert [e["event"] for e in events] == ["analysis", "analysis", "feedback"]
    baseline = json.loads((trained_tos / "meta.json").read_text())["monitoring_baseline"]
    summary = summarize(baselines={"tos": baseline})["tos"]
    assert summary["requests"] == 2 and summary["errors"] == {"too_short": 1}
    assert summary["feedback_agree_rate"] == 0.0 and summary["classified_units"] > 0


def test_repeat_analysis_is_served_from_cache(trained_tos, client):
    first = client.post("/analyze", data={"document_type": "tos", "text": TOS_TEXT}).json()
    second = client.post("/analyze", data={"document_type": "tos", "text": TOS_TEXT}).json()
    assert "cache_hit" not in first["diagnostics"] and second["diagnostics"]["cache_hit"]
    assert first["findings"] == second["findings"]


def test_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "What kind of document is this?" in page.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert client.get(asset).status_code == 200


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_ui_javascript():
    test_file = Path(__file__).parent / "js" / "app.test.mjs"
    result = subprocess.run(["node", "--test", str(test_file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
