"""Second round of features: the document-type router, recall-targeted thresholds,
oversampling, the joint model's loss mask, stale predictions, moving results between machines,
CORS for the browser extension, and sentence splitting around redaction markers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from clauseguard import pipeline
from clauseguard.config import OversampleSpec, load_module_config
from clauseguard.errors import InputError
from clauseguard.ingestion.segment import split_sentences
from clauseguard.models.runs import list_runs, read_json, resolve_run, select_run, selected_file
from clauseguard.models.thresholds import any_risk, tune_for_recall
from clauseguard.models.train import oversample_rare, train_run
from clauseguard.router import OTHER, Router, _pipeline, get_router, router_dir, windows
from clauseguard.web.app import create_app
from tests.conftest import CONTRACT, NEUTRAL, PRIVACY, SYNTHETIC
from tests.test_pipeline import TOS_TEXT

QUIET = lambda *_: None  # noqa: E731
NEWS = [
    "Heavy rain is expected across the region this weekend with local flooding.",
    "The city council approved a plan to build two new parks next spring.",
    "The home side came back from two goals down to win in the final minute.",
    "A new exhibition of landscape paintings opens at the national gallery.",
    "Researchers developed a cheaper way to recycle plastic bottles into fibres.",
    "Local bakeries report strong demand for seasonal bread this autumn.",
]
PRIVACY_TEXT = " ".join(s for ss in PRIVACY.values() for s in ss) * 3


@pytest.fixture
def router(home):
    """A small router trained on synthetic documents of each type, saved like the real one."""
    rng = np.random.default_rng(0)
    pools = {
        "tos": [s for ss in SYNTHETIC.values() for s in ss] + NEUTRAL,
        "privacy": [s for ss in PRIVACY.values() for s in ss],
        "contract": [s for ss in CONTRACT.values() for s in ss],
        OTHER: NEWS,
    }
    texts, labels = [], []
    for label, pool in pools.items():
        for _ in range(40):
            texts.append(" ".join(rng.choice(pool, size=12)))
            labels.append(label)
    pipe = _pipeline(4.0).fit(texts, labels)
    Router(pipe, list(pipe.classes_), 0.6, {"version": "router-test"}).save(router_dir())
    get_router.cache_clear()
    yield
    get_router.cache_clear()


# ---------------------------------------------------------------- router


def test_windows_cover_the_document():
    words = [f"w{i}" for i in range(2000)]
    parts = windows(" ".join(words), size=400, n=5)
    assert len(parts) == 5 and parts[0].split()[0] == "w0" and parts[-1].split()[-1] == "w1999"
    assert windows("a short text") == ["a short text"]


def test_router_detects_and_flags_unrelated_text(router):
    d = get_router().detect(TOS_TEXT)
    assert d.document_type == "tos" and d.confident and not d.unrelated
    news = get_router().detect(" ".join(NEWS * 3))
    assert news.unrelated and news.document_type is None and not news.confident
    assert set(d.as_dict()["probabilities"]) == {"tos", "privacy", "contract", OTHER}


def test_auto_type_uses_the_router(router, trained_tos):
    result = pipeline.analyze("auto", text=TOS_TEXT)
    assert result.document_type == "tos" and result.type_source == "router"
    assert result.detection["document_type"] == "tos"
    with pytest.raises(InputError) as exc:
        pipeline.analyze(None, text=" ".join(NEWS * 3))
    assert exc.value.code == "unrelated_document" and "detection" in exc.value.details


def test_chosen_type_that_looks_wrong_gets_a_warning(router, trained_tos):
    result = pipeline.analyze("tos", text=PRIVACY_TEXT)
    assert result.type_source == "user" and result.document_type == "tos"
    assert "looks more like a Privacy Policy" in result.warnings[0]


def test_auto_type_without_router_asks_the_user(trained_tos):
    with pytest.raises(InputError) as exc:
        pipeline.analyze("auto", text=TOS_TEXT)
    assert exc.value.code == "unknown_document_type"


def test_combined_analysis_runs_every_ready_module(trained_tos, trained_privacy):
    combined = pipeline.analyze_combined(["tos", "privacy"], text=TOS_TEXT)
    assert [r.document_type for r in combined.results] == ["tos", "privacy"]


def test_api_detect_auto_and_cors(router, trained_tos):
    client = TestClient(create_app())
    response = client.post("/detect", data={"text": TOS_TEXT})
    assert response.status_code == 200 and response.json()["document_type"] == "tos"
    response = client.post("/analyze", data={"text": TOS_TEXT})  # no type: auto
    assert response.status_code == 200 and response.json()["type_source"] == "router"
    response = client.post("/analyze", data={"document_type": "auto", "text": " ".join(NEWS * 3)})
    body = response.json()["error"]
    assert response.status_code == 422 and body["code"] == "unrelated_document"
    assert body["detection"]["unrelated"]
    origin = "chrome-extension://abcdefghijklmnop"
    preflight = client.options(
        "/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
    )
    assert preflight.headers["access-control-allow-origin"] == origin
    other = client.post(
        "/detect", data={"text": TOS_TEXT}, headers={"Origin": "https://evil.example"}
    )
    assert "access-control-allow-origin" not in other.headers
    assert client.get("/health").json()["router"] is True


def test_api_combined(trained_tos):
    client = TestClient(create_app())
    response = client.post("/analyze-combined", data={"document_types": "tos", "text": TOS_TEXT})
    assert response.status_code == 200 and len(response.json()["results"]) == 1


# ---------------------------------------------------------------- thresholds and sampling


def test_recall_target_raises_beta_until_reached():
    rng = np.random.default_rng(0)
    y = (rng.random((400, 3)) < 0.15).astype(int)
    p = np.clip(y * 0.45 + rng.random((400, 3)) * 0.6, 0, 1)
    risky = [True, True, True]
    th1, beta1, _ = tune_for_recall(y, p, risky, target=0.0)
    th2, beta2, tried = tune_for_recall(y, p, risky, target=0.99)
    assert beta1 == 1.0 and beta2 > 1.0
    assert any_risk(y, p, th2, risky)[0] >= any_risk(y, p, th1, risky)[0]
    # a precision floor stops the search before precision drops below it
    _, beta3, tried = tune_for_recall(y, p, risky, target=0.99, min_precision=0.99)
    assert tried[-1]["any_precision"] < 0.99 and beta3 == tried[-2]["beta"]


def test_oversampling_repeats_rare_labels_only():
    df = pd.DataFrame({"text": list("abcdefgh"), "labels": [["x:a"], ["x:a"], ["x:b"]] + [[]] * 5})
    out, factors = oversample_rare(
        df, ["x:a", "x:b"], OversampleSpec(min_positives=4, max_factor=3)
    )
    assert factors == {"x:a": 2, "x:b": 3}
    assert out["text"].tolist().count("a") == 2 and out["text"].tolist().count("c") == 3
    assert out["text"].tolist().count("d") == 1


def test_gpu_candidates_must_be_transformers():
    cfg = load_module_config("contract")
    assert set(cfg.session_models()) <= set(cfg.models)
    with pytest.raises(ValueError):
        type(cfg).model_validate({**cfg.model_dump(), "gpu_candidates": ["tfidf"]})


# ---------------------------------------------------------------- joint model


def test_joint_loss_mask_and_masked_pos_weights():
    from clauseguard.config import PosWeightSpec
    from clauseguard.joint import loss_mask
    from clauseguard.models.transformer import pos_weights

    labels = ["tos:a", "privacy:b", "contract:c"]
    mask = loss_mask(pd.Series(["tos", "contract"]), labels)
    assert mask.tolist() == [[1, 0, 0], [0, 0, 1]]
    y = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
    m = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    w = pos_weights(y, PosWeightSpec(mode="linear", max=50.0), m).tolist()
    assert w == [1.0, 1.0, 1.0]  # 1 positive vs 1 negative among the annotated rows


# ---------------------------------------------------------------- runs and results


def test_stale_predictions_are_not_reused(trained_tos, synthetic_tos):
    from clauseguard.data.units import save_units
    from clauseguard.evaluation.evaluate import evaluate_run, stored_predictions

    evaluate_run("tos", trained_tos.name, "validation")
    val = synthetic_tos[synthetic_tos["split"] == "validation"].reset_index(drop=True)
    labels = load_module_config("tos").label_ids
    assert stored_predictions(trained_tos, "validation", val, labels) is not None
    changed = synthetic_tos.copy()
    changed.loc[0, "text"] = changed.loc[0, "text"] + " (rebuilt)"
    save_units(changed, "tos")
    assert stored_predictions(trained_tos, "validation", val, labels) is None


def test_results_pack_and_import_keep_the_better_run(trained_tos, tmp_path):
    from clauseguard.results import import_results, pack_results

    first = trained_tos
    second = train_run("tos", "tfidf", seed=7, log=QUIET)
    select_run("tos", second.name)
    archive = pack_results(tmp_path / "out" / "results.tgz", ["tos"], log=QUIET)
    # pretend the second run happened on another machine
    import shutil

    shutil.rmtree(second)
    select_run("tos", first.name)
    summary = import_results(archive, log=QUIET)
    assert (second / "DONE").exists() and summary["runs_added"] == 1
    scores = {r["run_id"]: r["metrics_validation"]["macro_f1"] for r in list_runs("tos")}
    expected = second.name if scores[second.name] > scores[first.name] else first.name
    assert read_json(selected_file("tos"))["run_id"] == expected == resolve_run("tos").name
    assert summary["modules"]["tos"]["serving"] == expected


# ---------------------------------------------------------------- sentence splitting


def test_redaction_markers_do_not_split_sentences():
    text = (
        "The Distributor shall purchase at least [***] units per year. The fee is [***] per "
        "month, provided that it shall not exceed [***]. Payment is due within [*] days."
    )
    sentences = [s.text for s in split_sentences(text)]
    assert sentences == [
        "The Distributor shall purchase at least [***] units per year.",
        "The fee is [***] per month, provided that it shall not exceed [***].",
        "Payment is due within [*] days.",
    ]


# ---------------------------------------------------------------- screening


def test_screening_reads_fewer_units_with_the_same_flags(trained_tos, tmp_path, monkeypatch):
    """A transformer-like run (here: a second TF-IDF model posing as one) screened by the
    TF-IDF baseline: the analyzer reads fewer units and keeps the findings."""
    import json
    import shutil

    from clauseguard.evaluation.evaluate import evaluate_run
    from clauseguard.models.screening import SCREEN_DIR, SCREEN_FILE, screen_run
    from clauseguard.modules.base import Analyzer

    main = train_run("tos", "tfidf", seed=3, log=QUIET)
    meta = read_json(main / "meta.json")
    (main / "meta.json").write_text(json.dumps({**meta, "model_type": "transformer"}))
    evaluate_run("tos", main.name, "validation")
    select_run("tos", main.name)
    result = screen_run("tos", flags_kept=0.9, log=QUIET)
    if not result["used"]:  # the synthetic data may leave nothing to skip
        pytest.skip("screening would not save work on this synthetic data")
    assert (main / SCREEN_FILE).exists() and (main / SCREEN_DIR / "kind.json").exists()
    text = TOS_TEXT * 3
    monkeypatch.setenv("CLAUSEGUARD_SCREENING", "off")
    full = Analyzer("tos").analyze(pipeline.load_document(text=text))
    monkeypatch.setenv("CLAUSEGUARD_SCREENING", "on")
    screened = Analyzer("tos").analyze(pipeline.load_document(text=text))
    d = screened.diagnostics
    assert d["units_read_by_model"] <= d["classified_units"]
    assert {f.text for f in screened.findings} <= {f.text for f in full.findings}
    # packaging carries the screening model along (as a TF-IDF run again: no ONNX needed)
    from clauseguard.deploy import package_models

    (main / "meta.json").write_text(json.dumps(meta))

    package_models(tmp_path / "pkg", ["tos"], log=QUIET)
    assert (tmp_path / "pkg" / "runs" / "tos" / main.name / SCREEN_DIR / "kind.json").exists()
    shutil.rmtree(tmp_path / "pkg")
