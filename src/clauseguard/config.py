"""Typed access to ``configs/<module>.yaml`` and ``configs/taxonomy.yaml``."""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clauseguard.paths import configs_dir

DocumentType = Literal["tos", "privacy", "contract"]
DOCUMENT_TYPES: tuple[DocumentType, ...] = ("tos", "privacy", "contract")
Severity = Literal["high", "medium", "low", "protective", "info", "hidden"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelSpec(_Strict):
    id: str
    source: str  # the label's name in the source dataset


class ModelSpec(_Strict):
    type: Literal["majority", "tfidf", "transformer"]
    # TF-IDF + logistic regression
    ngram_max: int = 2
    min_df: int = 2
    max_features: int = 200_000
    c_grid: list[float] = Field(default_factory=lambda: [0.25, 1.0, 4.0])
    # transformer
    checkpoint: str | None = None
    lr: float = 3e-5
    epochs: int = 20
    batch_size: int = 16
    grad_accum: int = 1
    patience: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int | None = None  # defaults to the module's max_length
    # Training only: keep every labelled unit plus this many unlabelled units per labelled one.
    negative_ratio: float | None = None

    @model_validator(mode="after")
    def _needs_checkpoint(self) -> ModelSpec:
        if self.type == "transformer" and not self.checkpoint:
            raise ValueError("transformer models need a checkpoint")
        return self


class PosWeightSpec(_Strict):
    mode: Literal["none", "sqrt", "linear"] = "sqrt"
    max: float = 20.0


class SmokeSpec(_Strict):
    n_train: int = 200
    n_eval: int = 100
    epochs: int = 1
    max_length: int = 64
    batch_size: int = 8


class ModifierSpec(_Strict):
    """A sub-label that moves the severity of other labels when both are predicted."""

    label: str
    applies_to: list[str]
    change: Literal[-1, 1]


class ModuleConfig(_Strict):
    module: DocumentType
    display_name: str
    description: str
    unit_type: Literal["sentence", "paragraph"]
    labels: list[LabelSpec]
    text_style: Literal["none", "unfair_tos"] = "none"
    max_length: int
    min_unit_words: int = 3
    threshold_objective: Literal["f1", "f2"] = "f1"
    selection_metric: str = "macro_f1"
    pos_weight: PosWeightSpec = Field(default_factory=PosWeightSpec)
    risk_score_k: float
    seeds: list[int] = Field(default_factory=lambda: [13, 42, 87])
    models: dict[str, ModelSpec]
    smoke: SmokeSpec = Field(default_factory=SmokeSpec)
    # Paragraph segmentation (unit_type: paragraph)
    paragraph_min_words: int = 12
    paragraph_max_words: int = 180
    # Report: severity modifiers, statements a document should contain, merging of flags
    modifiers: list[ModifierSpec] = Field(default_factory=list)
    expected_statements: list[str] = Field(default_factory=list)
    missing_statement_weight: float = 1.0
    merge_adjacent: bool = False
    # Converter settings (split ratios, seeds, overlap rules), read by data/<module>.py
    data: dict = Field(default_factory=dict)

    @property
    def label_ids(self) -> list[str]:
        return [label.id for label in self.labels]

    @model_validator(mode="after")
    def _check_labels(self) -> ModuleConfig:
        ids = self.label_ids
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate label ids")
        bad = [i for i in ids if not i.startswith(f"{self.module}:")]
        if bad:
            raise ValueError(f"label ids must be namespaced with '{self.module}:': {bad}")
        referenced = set(self.expected_statements)
        for m in self.modifiers:
            referenced |= {m.label, *m.applies_to}
        unknown = referenced - set(ids)
        if unknown:
            raise ValueError(f"modifiers/expected_statements use unknown labels: {sorted(unknown)}")
        return self


class RiskGroup(_Strict):
    name: str
    description: str


class LabelInfo(_Strict):
    name: str
    risk_group: str
    severity: Severity
    definition: str
    template: str
    missing_message: str | None = None  # shown when an expected statement is absent


class Taxonomy(_Strict):
    severity_weights: dict[str, float]
    risk_bands: dict[str, float]  # band name -> lower bound of the score
    risk_groups: dict[str, RiskGroup]
    labels: dict[str, LabelInfo]

    @model_validator(mode="after")
    def _check_groups(self) -> Taxonomy:
        unknown = {i.risk_group for i in self.labels.values()} - set(self.risk_groups)
        if unknown:
            raise ValueError(f"unknown risk groups in taxonomy: {sorted(unknown)}")
        return self


def _read_yaml(name: str) -> dict:
    with open(configs_dir() / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_module_config(module: str) -> ModuleConfig:
    if module not in DOCUMENT_TYPES:
        raise ValueError(f"unknown module {module!r}; choose one of {', '.join(DOCUMENT_TYPES)}")
    path = configs_dir() / f"{module}.yaml"
    if not path.exists():
        raise ValueError(f"module {module!r} is not set up yet (missing {path.name})")
    return ModuleConfig.model_validate(_read_yaml(path.name))


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    return Taxonomy.model_validate(_read_yaml("taxonomy.yaml"))


def available_modules() -> list[DocumentType]:
    return [m for m in DOCUMENT_TYPES if (configs_dir() / f"{m}.yaml").exists()]
