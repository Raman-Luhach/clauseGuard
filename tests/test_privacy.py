"""Privacy module: OPP-115 converter (golden test on a hand-made corpus) and analyzer."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from clauseguard.data.privacy import prepare_privacy
from clauseguard.paths import processed_dir, raw_dir
from clauseguard.pipeline import analyze

SEGMENTS = [
    "<strong>Sharing</strong><br> We share your <b>email address</b> with advertising partners. "
    "We do not sell your data.",
    "You can delete your account at any time from the settings page of the app.",
]


def _attr(segment: str, text: str, value: str, *, wrong_offsets: bool = False) -> dict:
    start = segment.index(text) + (5 if wrong_offsets else 0)
    return {
        "selectedText": text,
        "startIndexInSegment": start,
        "endIndexInSegment": start + len(text),
        "value": value,
    }


def _no_span(value: str) -> dict:
    return {
        "selectedText": "null",
        "startIndexInSegment": -1,
        "endIndexInSegment": -1,
        "value": value,
    }


def _write_corpus(n_policies: int = 4) -> None:
    root = raw_dir("opp115") / "OPP-115"
    for folder in ("annotations", "sanitized_policies", "documentation/errant_span_indexes"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    s0, s1 = SEGMENTS
    third_ads = {
        "Purpose": _attr(s0, "advertising partners", "Advertising"),
        "Personal Information Type": _attr(s0, "email address", "Contact"),
    }
    third_plain = {"Purpose": _no_span("Unspecified")}
    negated = {"Does/Does Not": _attr(s0, "do not sell", "Does Not")}
    access = {"Access Type": _attr(s1, "delete your account", "Deactivate account")}
    access_bad_offsets = {
        "Access Type": _attr(s1, "delete your account", "Deactivate account", wrong_offsets=True)
    }
    # annotation id, annotator, segment, category, attributes
    practices = [
        ("1", "A", 0, "Third Party Sharing/Collection", third_ads),
        ("2", "B", 0, "Third Party Sharing/Collection", third_ads),
        ("3", "C", 0, "Third Party Sharing/Collection", third_plain),
        ("4", "A", 0, "First Party Collection/Use", negated),
        ("5", "B", 0, "First Party Collection/Use", negated),
        ("6", "A", 0, "Data Security", {"Security Measure": _no_span("Generic")}),
        ("7", "A", 1, "User Access, Edit and Deletion", access),
        ("8", "B", 1, "User Access, Edit and Deletion", access),
        ("9", "C", 1, "User Access, Edit and Deletion", access_bad_offsets),
    ]
    for i in range(n_policies):
        name = f"{i + 1}_site{i}.com"
        (root / "sanitized_policies" / f"{name}.html").write_text("|||".join(SEGMENTS))
        with open(root / "annotations" / f"{name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for aid, annotator, sid, category, attrs in practices:
                writer.writerow(
                    [
                        f"{i}{aid}",
                        "batch",
                        annotator,
                        i + 1,
                        sid,
                        category,
                        json.dumps(attrs),
                        "http://example.com",
                        "1/1/16",
                    ]
                )


@pytest.fixture
def opp_units(home):
    _write_corpus()
    return prepare_privacy()


def test_majority_vote_and_sub_labels(opp_units):
    first = opp_units[opp_units["unit_id"] == "1_site0.com:0"].iloc[0]
    assert first["labels"] == [
        "privacy:first_party_collection",
        "privacy:third_party_sharing",
        "privacy:sharing_ads_marketing",  # 2 of 3 annotators said Advertising
        "privacy:negated_practice",
    ]  # Data Security had a single vote, so it is not a label
    second = opp_units[opp_units["unit_id"] == "1_site0.com:1"].iloc[0]
    assert second["labels"] == ["privacy:user_access"]
    assert json.loads(second["meta"])["votes"] == {"privacy:user_access": 3}


def test_html_is_removed_and_offsets_cover_the_policy_text(opp_units):
    policy = opp_units[opp_units["doc_id"] == "1_site0.com"]
    assert "<" not in "".join(policy["text"])
    assert policy["text"].iloc[0].startswith("Sharing\nWe share your email address")
    assert list(policy["char_start"]) == [0, len(policy["text"].iloc[0]) + 2]


def test_every_policy_is_in_exactly_one_split(opp_units, home):
    assert opp_units.groupby("doc_id")["split"].nunique().max() == 1
    saved = json.loads((home / "data" / "splits" / "privacy.json").read_text())
    assert set(saved["assignment"]) == set(opp_units["doc_id"])


def test_spans_reach_the_right_sentences(opp_units):
    stats = json.loads((processed_dir("privacy") / "conversion_stats.json").read_text())
    assert stats.get("spans_unmapped", 0) == 0
    assert stats["spans_found_by_search"] == 4  # annotator C's wrong offsets, 4 policies
    sentences = pd.read_parquet(processed_dir("privacy") / "sentences.parquet")
    by_text = {
        t: list(labs)
        for t, labs in zip(sentences["text"], sentences["labels"], strict=True)
        if "site" not in t
    }
    assert by_text["We share your email address with advertising partners."] == [
        "privacy:third_party_sharing",
        "privacy:sharing_ads_marketing",
    ]
    assert by_text["We do not sell your data."] == [
        "privacy:first_party_collection",
        "privacy:negated_practice",
    ]


POLICY = """ACME PRIVACY POLICY

What we collect
We collect your name, email address and phone number when you register. We also record your
location when you use the app.

How we share information
We share your browsing data with advertising networks to show you targeted ads. Advertising
partners receive your device identifiers for marketing campaigns.

Your choices
You can opt out of marketing emails at any time by clicking unsubscribe in any message.

Changes
We may update this policy and will post the changes on this page whenever we make them.
"""


def test_privacy_analysis_uses_paragraphs_modifiers_and_protections(trained_privacy):
    result = analyze("privacy", text=POLICY)
    assert all(f.unit_type == "paragraph" for f in result.findings)
    sharing = next(f for f in result.findings if f.labels[0].label == "privacy:third_party_sharing")
    assert sharing.severity == "high"  # medium raised by the advertising sub-label
    assert "advertising or marketing" in sharing.explanation
    change = next(f for f in result.findings if f.labels[0].label == "privacy:policy_change")
    assert change.severity == "medium"  # low raised: changes only posted on the website
    assert "Your choices (opt-in / opt-out)" in result.protections_found
    assert any("protected" in m for m in result.protections_missing)
    assert any("access" in m.lower() or "delete" in m.lower() for m in result.protections_missing)
    for finding in result.findings:
        assert result.text[finding.char_start : finding.char_end] == finding.text
        assert all(lab.severity != "hidden" for lab in finding.labels)
