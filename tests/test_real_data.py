"""Regression checks on the real OPP-115 and CUAD unit tables (skipped until `make data`)."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

from clauseguard.config import load_module_config
from clauseguard.data.units import label_matrix, load_units
from clauseguard.paths import processed_dir, splits_dir

pytestmark = pytest.mark.slow


def _needs(module: str) -> None:
    if not (processed_dir(module) / "units.parquet").exists():
        pytest.skip(f"run `clauseguard data prepare {module}` first")


def _stats(module: str) -> dict:
    return json.loads((processed_dir(module) / "conversion_stats.json").read_text())


def test_privacy_conversion():
    _needs("privacy")
    units, stats = load_units("privacy"), _stats("privacy")
    assert len(units) == 3792 and units["doc_id"].nunique() == 115
    assert units.drop_duplicates("doc_id")["split"].value_counts().to_dict() == {
        "train": 80,
        "validation": 17,
        "test": 18,
    }
    expected = {  # 2-of-3 majority counts, verified independently from the raw CSVs
        "privacy:third_party_sharing": 945,
        "privacy:first_party_collection": 1209,
        "privacy:data_retention": 78,
        "privacy:do_not_track": 31,
        "privacy:sharing_ads_marketing": 265,
        "privacy:negated_practice": 160,
        "privacy:change_without_personal_notice": 108,
    }
    counts = Counter(label for labels in units["labels"] for label in labels)
    assert {k: counts[k] for k in expected} == expected
    assert stats.get("spans_unmapped", 0) == 0 and stats.get("empty_segments", 0) == 0
    assert not units["text"].str.contains(r"<(?:br|b|p|strong|li)\b", regex=True).any()
    saved = json.loads((splits_dir() / "privacy.json").read_text())["assignment"]
    assert units.drop_duplicates("doc_id").set_index("doc_id")["split"].to_dict() == saved


def test_contract_conversion():
    _needs("contract")
    cfg = load_module_config("contract")
    units, stats = load_units("contract"), _stats("contract")
    docs = units.drop_duplicates("doc_id")["split"].value_counts().to_dict()
    assert docs == {"train": 367, "validation": 41, "test": 102}
    covered = sum(v for k, v in stats.items() if k.startswith("covered:"))
    total = sum(v for k, v in stats.items() if k.startswith("spans:"))
    assert covered / total >= 0.99
    assert stats.get("offset_mismatch", 0) == 0
    assert units["exclude_from_training"].sum() == stats["ambiguous_sentences"] < 200
    for split in ("train", "validation", "test"):
        y = label_matrix(units.loc[units["split"] == split, "labels"], cfg.label_ids)
        assert y.sum(axis=0).min() > 0, f"a label has no examples in {split}"
    spans = pd.read_parquet(processed_dir("contract") / "spans.parquet")
    assert len(spans) == total and spans["split"].notna().all()
