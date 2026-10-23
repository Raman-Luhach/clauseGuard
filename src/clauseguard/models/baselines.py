"""Majority and TF-IDF + logistic regression baselines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from clauseguard.config import ModelSpec
from clauseguard.evaluation.metrics import macro_pr_auc
from clauseguard.models.base import MultiLabelModel


class MajorityModel(MultiLabelModel):
    """Predicts each label's training frequency for every input, i.e. "never risky"."""

    kind = "majority"

    def __init__(self, labels: list[str], prior: np.ndarray | None = None) -> None:
        self.labels = labels
        self.prior = prior if prior is not None else np.zeros(len(labels))

    def fit(self, y: np.ndarray) -> MajorityModel:
        self.prior = y.mean(axis=0)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.tile(self.prior, (len(texts), 1))

    def save(self, path: Path) -> None:
        self._write_kind(path)
        (path / "prior.json").write_text(json.dumps(self.prior.tolist()))

    @classmethod
    def load(cls, path: Path) -> MajorityModel:
        kind = json.loads((path / "kind.json").read_text())
        prior = np.array(json.loads((path / "prior.json").read_text()))
        return cls(kind["labels"], prior)


class TfidfModel(MultiLabelModel):
    """Word 1-2 gram TF-IDF with one balanced logistic regression per label.

    The regularisation strength C is picked on the validation set by macro PR-AUC.
    """

    kind = "tfidf"

    def __init__(self, labels: list[str], vectorizer=None, classifier=None) -> None:
        self.labels = labels
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(
        self,
        spec: ModelSpec,
        texts: Sequence[str],
        y: np.ndarray,
        val_texts: Sequence[str],
        val_y: np.ndarray,
        seed: int,
    ) -> dict:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, spec.ngram_max),
            min_df=spec.min_df,
            max_features=spec.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        x = self.vectorizer.fit_transform(texts)
        x_val = self.vectorizer.transform(val_texts)
        scores: dict[str, float] = {}
        best = None
        for c in spec.c_grid:
            clf = OneVsRestClassifier(
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            )
            # Sequential on purpose: process workers hand liblinear read-only memmaps, and
            # thread workers race on the shared sparse matrix (it is sorted in place).
            clf.fit(x, y)
            score = macro_pr_auc(val_y, clf.predict_proba(x_val))
            scores[str(c)] = score
            if best is None or score > best[0]:
                best = (score, c, clf)
        assert best is not None
        self.classifier = best[2]
        return {
            "chosen_C": best[1],
            "val_macro_pr_auc_by_C": scores,
            "vocabulary": len(self.vectorizer.vocabulary_),
        }

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, len(self.labels)))
        return self.classifier.predict_proba(self.vectorizer.transform(texts))

    def save(self, path: Path) -> None:
        self._write_kind(path)
        joblib.dump(
            {"vectorizer": self.vectorizer, "classifier": self.classifier}, path / "tfidf.joblib"
        )

    @classmethod
    def load(cls, path: Path) -> TfidfModel:
        kind = json.loads((path / "kind.json").read_text())
        parts = joblib.load(path / "tfidf.joblib")
        return cls(kind["labels"], parts["vectorizer"], parts["classifier"])
