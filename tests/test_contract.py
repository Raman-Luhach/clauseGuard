"""Contract module: span-to-sentence conversion (golden tests) and the analyzer."""

from __future__ import annotations

import json

import pandas as pd

from clauseguard.data.contract import label_sentences, prepare_contract
from clauseguard.ingestion.segment import Span
from clauseguard.paths import processed_dir, raw_dir
from clauseguard.pipeline import analyze


def _sentences(*lengths: int) -> list[Span]:
    spans, start = [], 0
    for n in lengths:
        spans.append(Span(start, start + n, "x" * n))
        start += n + 1
    return spans


def test_overlap_rule():
    s = _sentences(100, 100, 100)  # [0,100) [101,201) [202,302)
    positive, ambiguous, covered = label_sentences(
        s,
        [
            ("short_inside", 10, 30),  # short span inside sentence 0
            ("multi", 50, 260),  # half of s0, all of s1, 58 characters of s2
            ("spill", 105, 205),  # s1 + 3 characters of s2
        ],
        threshold=0.5,
    )
    assert positive[0] == {"short_inside", "multi"}
    assert positive[1] == {"multi", "spill"}
    assert positive[2] == {"multi"}
    assert ambiguous[2] == {"spill"} and ambiguous[1] == set()
    assert covered == [True, True, True]


CONTRACT = (
    "DISTRIBUTION AGREEMENT\n\n"
    "1. Term. This Agreement starts on the Effective Date and continues for two years.\n"
    "2. Governing Law. This Agreement shall be governed by the laws of the\n"
    "State of Delaware.\n"
    "3. Assignment. Neither party may assign this Agreement without the prior written consent "
    "of the other party. Any attempted assignment is void.\n"
)


GOVERNING_LAW = "This Agreement shall be governed by the laws of the\nState of Delaware."


def _write_cuad(titles_train: list[str], titles_test: list[str]) -> None:
    root = raw_dir("cuad")
    root.mkdir(parents=True, exist_ok=True)

    def doc(title: str) -> dict:
        def qa(category: str, text: str) -> dict:
            return {
                "id": f"{title}__{category}",
                "question": "?",
                "is_impossible": False,
                "answers": [{"text": text, "answer_start": CONTRACT.index(text)}],
            }

        return {
            "title": title,
            "paragraphs": [
                {
                    "context": CONTRACT,
                    "qas": [
                        qa("Governing Law", GOVERNING_LAW),
                        qa(
                            "Anti-Assignment",
                            "Neither party may assign this Agreement without the prior",
                        ),
                        qa("Document Name", "DISTRIBUTION AGREEMENT"),  # not a selected category
                    ],
                }
            ],
        }

    everything = [doc(t) for t in titles_train + titles_test]
    (root / "CUADv1.json").write_text(json.dumps({"version": "v1", "data": everything}))
    (root / "train_separate_questions.json").write_text(
        json.dumps({"data": [doc(t) for t in titles_train]})
    )
    (root / "test.json").write_text(json.dumps({"data": [doc(t) for t in titles_test]}))


def test_cuad_conversion(home):
    train, test = [f"train{i}" for i in range(10)], ["test0", "test1"]
    _write_cuad(train, test)
    units = prepare_contract()
    first = units[units["doc_id"] == "train0"]
    labels = {t: list(labs) for t, labs in zip(first["text"], first["labels"], strict=True)}
    # the hard-wrapped span was unwrapped by cleaning and still lands on its sentence
    assert labels[
        "2. Governing Law. This Agreement shall be governed by the laws of the State of Delaware."
    ] == ["contract:governing_law"]
    assert labels[
        "3. Assignment. Neither party may assign this Agreement without the prior written "
        "consent of the other party."
    ] == ["contract:anti_assignment"]
    assert labels["Any attempted assignment is void."] == []
    assert "DISTRIBUTION AGREEMENT" not in labels  # headings are not units
    assert set(units.loc[units["doc_id"].isin(test), "split"]) == {"test"}
    assert set(units.loc[units["doc_id"].isin(train), "split"]) == {"train", "validation"}
    for text, start, end in zip(first["text"], first["char_start"], first["char_end"], strict=True):
        assert end - start == len(text)
    spans = pd.read_parquet(processed_dir("contract") / "spans.parquet")
    assert set(spans["label"]) == {"contract:governing_law", "contract:anti_assignment"}
    stats = json.loads((processed_dir("contract") / "conversion_stats.json").read_text())
    assert stats["covered:contract:governing_law"] == stats["spans:contract:governing_law"] == 12


AGREEMENT = (
    "This Distribution Agreement is made between Acme Inc. and Beta LLC. "
    "The parties have executed this Agreement as of the Effective Date. "
    "This Agreement shall be governed by the laws of the State of Delaware. "
    "This Agreement is governed by and construed under the laws of New York. "
    "The Distributor shall deliver monthly sales reports by email to the Company. "
    "Neither party may assign this Agreement without the prior written consent of the other."
)


def test_adjacent_flags_are_merged(trained_contract):
    result = analyze("contract", text=AGREEMENT)
    law = [f for f in result.findings if f.labels[0].label == "contract:governing_law"]
    assert len(law) == 1 and law[0].n_units == 2
    assert law[0].text.startswith("This Agreement shall be governed") and law[0].text.endswith(
        "laws of New York."
    )
    assert result.text[law[0].char_start : law[0].char_end] == law[0].text
    assert any(f.labels[0].label == "contract:anti_assignment" for f in result.findings)
