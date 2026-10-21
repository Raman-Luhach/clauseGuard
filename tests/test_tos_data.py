"""Checks against the real UNFAIR-ToS / CLAUDETTE files (skipped when not downloaded)."""

from __future__ import annotations

import pandas as pd
import pytest

from clauseguard.config import load_module_config
from clauseguard.data.tos import EXPECTED_ROWS, claudette_label_agreement, prepare_tos
from clauseguard.ingestion.normalize import unfair_tos_style
from clauseguard.paths import raw_dir

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (raw_dir("unfair_tos") / ".complete").exists()
        or not (raw_dir("claudette") / ".complete").exists(),
        reason="run `clauseguard data download tos` first",
    ),
]


@pytest.fixture(scope="module")
def units() -> pd.DataFrame:
    return prepare_tos()


def test_row_counts_and_splits(units):
    assert units["split"].value_counts().to_dict() == EXPECTED_ROWS
    assert units.groupby("doc_id")["split"].nunique().max() == 1


def test_every_sentence_is_matched_to_one_of_50_documents(units):
    assert (units["doc_id"] != "unknown").all()
    assert units["doc_id"].nunique() == 50
    assert units.drop_duplicates("doc_id")["split"].value_counts().to_dict() == {
        "train": 30,
        "validation": 10,
        "test": 10,
    }


def test_labels_agree_with_original_claudette_files(units):
    agreement = claudette_label_agreement(
        units, raw_dir("claudette") / "ToS", load_module_config("tos").label_ids
    )
    assert len(agreement) == 8 and min(agreement.values()) == 1.0


def test_normaliser_barely_changes_training_text():
    raw = pd.concat(pd.read_parquet(raw_dir("unfair_tos") / f"{s}.parquet") for s in EXPECTED_ROWS)
    changed = sum(unfair_tos_style(t) != " ".join(t.split()) for t in raw["text"])
    assert changed / len(raw) < 0.005
