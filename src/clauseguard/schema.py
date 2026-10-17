"""The output format shared by every module (CLI JSON, REST API, web UI)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from clauseguard.config import DocumentType, Severity

DISCLAIMER = (
    "ClauseGuard highlights clauses that may work against you. It is not legal advice; check "
    "important decisions with a qualified professional."
)


class LabelScore(BaseModel):
    label: str
    name: str
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity  # after modifiers, e.g. sharing for advertising -> high


class Finding(BaseModel):
    module: DocumentType
    unit_type: Literal["sentence", "paragraph"]
    unit_index: int
    n_units: int = 1  # consecutive units merged into this finding (contracts)
    text: str
    char_start: int  # offsets into AnalysisResult.text
    char_end: int
    page: int | None = None  # 1-based, PDFs only
    labels: list[LabelScore]  # labels above threshold, highest score first
    risk_group: str
    risk_group_name: str
    severity: Severity
    explanation: str
    explanation_source: Literal["template", "llm"] = "template"
    _notes: list[str] = PrivateAttr(default_factory=list)  # modifier notes, kept for the LLM


class ModelInfo(BaseModel):
    run_id: str
    model: str
    trained_at: str | None = None


class AnalysisResult(BaseModel):
    document_type: DocumentType
    document_type_name: str
    model: ModelInfo
    n_units: int
    n_words: int
    n_pages: int | None = None
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_band: Literal["low", "medium", "high"] | None = None
    severity_counts: dict[str, int]
    findings: list[Finding]
    protections_found: list[str] = Field(default_factory=list)
    protections_missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    # For monitoring: units classified, flags per label (before merging), borderline flags
    diagnostics: dict = Field(default_factory=dict)
    text: str  # the cleaned document text that the offsets refer to
    disclaimer: str = DISCLAIMER
