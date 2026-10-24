from __future__ import annotations

import math

import numpy as np
import pytest

from clauseguard.config import ModelSpec, available_modules, load_module_config, load_taxonomy
from clauseguard.evaluation.error_analysis import NONE, confusion_pairs
from clauseguard.evaluation.metrics import compute_metrics
from clauseguard.models.base import load_model
from clauseguard.models.baselines import MajorityModel, TfidfModel
from clauseguard.models.thresholds import apply_thresholds, best_threshold, tune_thresholds
from clauseguard.scoring.risk_score import calibrate_k, raw_risk, risk_band, score_from_raw

# ---------------------------------------------------------------- config / taxonomy


def test_every_module_label_has_a_complete_taxonomy_entry():
    taxonomy = load_taxonomy()
    for module in available_modules():
        for label in load_module_config(module).label_ids:
            info = taxonomy.labels[label]
            assert info.template.strip() and info.definition.strip()


def test_unknown_module_is_rejected():
    with pytest.raises(ValueError, match="unknown module"):
        load_module_config("router")


# ---------------------------------------------------------------- thresholds


def test_best_threshold_separable():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, f = best_threshold(y, p)
    assert 0.4 < threshold <= 0.6 and f == 1.0
    assert apply_thresholds(p[:, None], [threshold]).ravel().tolist() == [0, 0, 1, 1]


def test_f2_prefers_recall():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
    t1, _ = best_threshold(y, p, beta=1.0)
    t2, _ = best_threshold(y, p, beta=2.0)
    assert t2 <= t1


def test_label_without_positives_keeps_default():
    threshold, f = best_threshold(np.zeros(5), np.linspace(0, 1, 5))
    assert threshold == 0.5 and math.isnan(f)
    assert tune_thresholds(np.zeros((3, 2)), np.full((3, 2), 0.7)) == [0.5, 0.5]


# ---------------------------------------------------------------- metrics


def test_metrics_perfect_and_lexglue_style():
    y = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    p = y.astype(float) * 0.9 + 0.05
    m = compute_metrics(y, p, [0.5, 0.5], ["a", "b"], lexglue=True)
    assert m["micro"]["f1"] == 1.0 and m["macro_f1"] == 1.0 and m["any_risk"]["recall"] == 1.0
    assert m["lexglue"]["micro_f1"] == 1.0
    assert m["per_label"]["a"]["support"] == 2


def test_macro_ignores_labels_absent_from_eval_data():
    y = np.array([[1, 0], [0, 0]])
    p = np.array([[0.9, 0.1], [0.1, 0.1]])
    m = compute_metrics(y, p, [0.5, 0.5], ["a", "b"])
    assert m["macro_labels"] == 1 and m["macro_f1"] == 1.0


def test_any_risk_uses_only_risky_labels():
    y = np.array([[0, 1], [0, 0]])
    p = np.array([[0.1, 0.9], [0.1, 0.1]])
    m = compute_metrics(y, p, [0.5, 0.5], ["risky", "protective"], risky=[True, False])
    assert math.isnan(m["any_risk"]["pr_auc"]) and m["any_risk"]["recall"] == 0.0


def test_confusion_pairs():
    y = np.array([[1, 0], [0, 0], [0, 1]])
    pred = np.array([[0, 1], [1, 0], [0, 0]])
    pairs = {
        (r.predicted, r.gold): r.count for r in confusion_pairs(y, pred, ["a", "b"]).itertuples()
    }
    assert pairs == {("b", "a"): 1, ("a", NONE): 1, (NONE, "b"): 1}


# ---------------------------------------------------------------- risk score


def test_raw_risk_counts_each_label_once_with_its_best_score():
    raw = raw_risk(
        [
            ("tos:arbitration", "high", 0.5),
            ("tos:arbitration", "high", 0.9),
            ("tos:choice_of_law", "low", 1.0),
            ("x", "protective", 1.0),
        ]
    )
    assert raw == pytest.approx(3 * 0.9 + 1 * 1.0)


def test_score_and_bands():
    k = calibrate_k([2.0, 4.0, 6.0], default=8.0)
    assert score_from_raw(4.0, k) == pytest.approx(50.0, abs=0.1)
    assert score_from_raw(0.0, k) == 0.0
    assert score_from_raw(1e9, k) == 100.0
    assert [risk_band(s) for s in (0, 33.9, 34, 66.9, 67, 100)] == [
        "low",
        "low",
        "medium",
        "medium",
        "high",
        "high",
    ]
    assert calibrate_k([1.0], default=8.0) == 8.0  # too few documents to calibrate


# ---------------------------------------------------------------- baselines save/load


def test_tfidf_and_majority_roundtrip(tmp_path):
    texts = [
        "we may terminate your account",
        "binding arbitration applies",
        "hello there friend",
    ] * 5
    y = np.array([[1, 0], [0, 1], [0, 0]] * 5)
    spec = ModelSpec(type="tfidf", min_df=1, c_grid=[1.0, 4.0])
    model = TfidfModel(["a", "b"])
    info = model.fit(spec, texts, y, texts, y, seed=0)
    assert info["chosen_C"] in (1.0, 4.0)
    model.save(tmp_path / "tfidf")
    loaded = load_model(tmp_path / "tfidf")
    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts))
    assert loaded.predict_proba([]).shape == (0, 2)

    majority = MajorityModel(["a", "b"]).fit(y)
    majority.save(tmp_path / "majority")
    np.testing.assert_allclose(
        load_model(tmp_path / "majority").predict_proba(["x"]), [[1 / 3, 1 / 3]]
    )
