"""Shared building blocks added for the privacy and contract modules."""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from clauseguard.config import load_module_config, load_taxonomy
from clauseguard.data import download as dl
from clauseguard.data.splits import load_or_create_split, stratified_group_split
from clauseguard.data.units import label_matrix, save_units
from clauseguard.evaluation.metrics import clause_recall, document_presence
from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.segment import split_paragraphs
from clauseguard.models.train import sample_negatives, train_run
from clauseguard.scoring.rules import effective_severity, statements
from clauseguard.textmap import strip_html
from tests.conftest import synthetic_tos_units

# ---------------------------------------------------------------- splits


def _docs(n: int = 40) -> dict[str, set[str]]:
    rare = {f"d{i}" for i in range(0, n, 10)}  # 4 documents carry the rare label
    return {f"d{i}": ({"common"} | ({"rare"} if f"d{i}" in rare else set())) for i in range(n)}


def test_stratified_split_is_deterministic_and_spreads_rare_labels():
    docs = _docs()
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    a = stratified_group_split(docs, ratios, seed=1)
    assert a == stratified_group_split(docs, ratios, seed=1)
    assert set(a) == set(docs)
    assert Counter(a.values()) == {"train": 20, "validation": 10, "test": 10}
    rare_splits = Counter(a[d] for d, labels in docs.items() if "rare" in labels)
    assert set(rare_splits) == {"train", "validation", "test"}


def test_saved_split_is_reused_and_guarded(home):
    docs = _docs()
    first = load_or_create_split("demo", docs, {"train": 0.8, "test": 0.2}, seed=1)
    assert load_or_create_split("demo", docs, {"train": 0.5, "test": 0.5}, seed=9) == first
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_split("demo", {**docs, "new": set()}, {"train": 0.8, "test": 0.2}, seed=1)


# ---------------------------------------------------------------- paragraphs and maps


def test_paragraphs_follow_blank_lines_and_headings():
    text = (
        "Information We Collect\nWe collect your name and email address when you sign up for an "
        "account with us.\n\nHow We Share Information\nWe share data with partners who help us "
        "run the service and with advertisers.\n\nContact\nEmail us."
    )
    paragraphs = split_paragraphs(text, min_words=8, max_words=180)
    assert [p.text.split("\n")[0] for p in paragraphs] == [
        "Information We Collect",
        "How We Share Information",
    ]
    assert paragraphs[-1].text.endswith("Contact\nEmail us.")  # too short alone: merged back
    for p in paragraphs:
        assert text[p.start : p.end] == p.text


def test_long_pdf_style_section_is_cut_at_sentences():
    sentence = "We may share your information with carefully selected partners for analytics. "
    text = "Sharing\n" + "\n".join([sentence.strip()] * 30)  # no blank lines, like PDF text
    paragraphs = split_paragraphs(text, min_words=8, max_words=60)
    assert len(paragraphs) > 1
    assert all(len(p.text.split()) <= 60 for p in paragraphs)
    assert all(p.text.rstrip().endswith(".") for p in paragraphs)


def test_composed_maps_point_back_into_raw_html():
    html = "Intro<br>We share your <b>email\naddress</b> with partners."
    stripped = strip_html(html)
    composed = stripped.then(clean_text(stripped.text))
    start = html.index("email")
    s, e = composed.from_source_span(start, html.index("</b>"))
    assert composed.text[s:e] == "email address"
    assert composed.to_source_span(s, e) == (start, html.index("</b>"))


# ---------------------------------------------------------------- severity rules


def test_privacy_modifiers_move_severity_one_step():
    cfg, tax = load_module_config("privacy"), load_taxonomy()
    sharing = "privacy:third_party_sharing"

    def sev(label: str, *others: str) -> str:
        return effective_severity(cfg, tax, label, {label, *others})[0]

    assert sev(sharing) == "medium"
    assert sev(sharing, "privacy:sharing_ads_marketing") == "high"
    assert sev(sharing, "privacy:sharing_ads_marketing", "privacy:negated_practice") == "medium"
    assert sev("privacy:first_party_collection", "privacy:negated_practice") == "info"
    assert sev("privacy:policy_change", "privacy:change_without_personal_notice") == "medium"
    assert sev("privacy:user_choice", "privacy:negated_practice") == "protective"
    _, notes = effective_severity(cfg, tax, sharing, {sharing, "privacy:sharing_ads_marketing"})
    assert notes == ["The sharing appears to be for advertising or marketing."]


def test_expected_statements():
    cfg, tax = load_module_config("privacy"), load_taxonomy()
    found, missing, extra = statements(cfg, tax, {"privacy:user_choice", "privacy:other"})
    assert found == ["Your choices (opt-in / opt-out)"]
    assert len(missing) == 3 and extra == 3.0
    assert statements(load_module_config("tos"), tax, set()) == ([], [], 0.0)


# ---------------------------------------------------------------- metrics


def test_document_presence():
    y = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    pred = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    m = document_presence(["a", "a", "b", "b"], y, pred, ["x", "y"])
    assert m["n_documents"] == 2
    assert m["per_label"]["x"]["recall"] == 1.0  # doc a has x somewhere, and it was found
    assert m["per_label"]["y"]["precision"] == 1.0 and m["per_label"]["y"]["recall"] == 1.0


def test_clause_recall():
    units = pd.DataFrame(
        {"doc_id": ["a", "a", "b"], "char_start": [0, 50, 0], "char_end": [49, 99, 49]}
    )
    pred = np.array([[0], [1], [0]])
    spans = pd.DataFrame(
        {"doc_id": ["a", "a", "b"], "label": ["x"] * 3, "start": [10, 40, 5], "end": [20, 60, 10]}
    )
    m = clause_recall(spans, units, pred, ["x"])
    assert m["per_label"]["x"]["recall"] == pytest.approx(1 / 3) and m["n_spans"] == 3


# ---------------------------------------------------------------- training data selection


def test_sample_negatives_keeps_every_positive():
    df = synthetic_tos_units()
    ids = load_module_config("tos").label_ids
    sampled = sample_negatives(df, ids, ratio=0.5, seed=0)
    y_all = label_matrix(df["labels"], ids).any(axis=1)
    y_s = label_matrix(sampled["labels"], ids).any(axis=1)
    assert y_s.sum() == y_all.sum() and (~y_s).sum() == round(0.5 * y_all.sum())


def test_excluded_units_are_not_trained_on(home):
    df = synthetic_tos_units()
    excluded = df.index[df["split"] == "train"][:10]
    df.loc[excluded, "exclude_from_training"] = True
    save_units(df, "tos")
    run = train_run("tos", "tfidf", log=lambda *_: None)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["n_train_excluded"] == 10
    assert meta["n_train"] == int((df["split"] == "train").sum()) - 10


def test_cross_validation_on_synthetic_data(synthetic_tos):
    from clauseguard.evaluation.cv import cross_validate

    report = cross_validate("tos", "tfidf", folds=3, seed=0, log=lambda *_: None)
    assert len(report["folds_detail"]) == 3
    assert report["summary"]["macro F1"]["mean"] > 0.9
    with pytest.raises(ValueError, match="TF-IDF"):
        cross_validate("tos", "legal-bert")


def test_extraction_skips_listed_folders(home, tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("corpus/keep/a.txt", "a")
        zf.writestr("corpus/unused/b.txt", "b")
    src = tmp_path / "c.zip"
    src.write_bytes(buf.getvalue())
    import hashlib

    manifest = {
        "demo": {
            "description": "d",
            "license": "l",
            "files": [
                {
                    "url": src.as_uri(),
                    "filename": "c.zip",
                    "extract": True,
                    "skip": ["corpus/unused/"],
                    "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
                }
            ],
        }
    }
    monkeypatch.setattr(dl, "load_manifest", lambda: manifest)
    target = dl.download("demo")
    assert (target / "corpus" / "keep" / "a.txt").exists()
    assert not (target / "corpus" / "unused").exists()
