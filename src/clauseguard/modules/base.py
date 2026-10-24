"""The analyzer: segment a document, classify each unit, build findings, score.

One class serves every module; differences live in ``configs/<module>.yaml``: the unit type
(sentence or paragraph), the text style the model expects, severity modifiers (privacy
sub-labels), statements a document should contain (privacy protections), and whether
adjacent flags with the same label are merged (contracts, whose clauses span sentences).
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from clauseguard.config import load_module_config, load_taxonomy
from clauseguard.explain.llm import Clause, Explainer
from clauseguard.explain.templates import compose_explanation, template_explanation
from clauseguard.ingestion.loader import Document
from clauseguard.ingestion.normalize import unfair_tos_style
from clauseguard.ingestion.segment import Span, is_classifiable, split_paragraphs, split_sentences
from clauseguard.models.base import load_model, serving_model_dir
from clauseguard.models.runs import read_json, resolve_run
from clauseguard.monitoring import BORDERLINE_MARGIN
from clauseguard.schema import AnalysisResult, Finding, LabelScore, ModelInfo
from clauseguard.scoring.risk_score import raw_risk, risk_band, score_from_raw
from clauseguard.scoring.rules import effective_severity, statements

SHOWN_SEVERITIES = ("high", "medium", "low", "protective", "info")


class Analyzer:
    def __init__(
        self,
        module: str,
        run_id: str | None = None,
        device: str = "auto",
        explainer: Explainer | None = None,
    ) -> None:
        self.cfg = load_module_config(module)
        self.explainer = explainer
        self.taxonomy = load_taxonomy()
        self.run_path: Path = resolve_run(module, run_id, self.cfg.selection_metric)
        self.meta = read_json(self.run_path / "meta.json")
        self.model = load_model(serving_model_dir(self.run_path), device=device)
        by_label = read_json(self.run_path / "thresholds.json")
        self.thresholds = np.array([by_label[label] for label in self.cfg.label_ids])
        self.risk_k = float(self.meta.get("risk_score_k") or self.cfg.risk_score_k)

    def segment(self, doc: Document) -> list[Span]:
        if self.cfg.unit_type == "paragraph":
            return split_paragraphs(
                doc.text,
                min_words=self.cfg.paragraph_min_words,
                max_words=self.cfg.paragraph_max_words,
            )
        return split_sentences(doc.text)

    def model_text(self, text: str) -> str:
        return unfair_tos_style(text) if self.cfg.text_style == "unfair_tos" else text

    def _finding(
        self, index: int, span: Span, scores: np.ndarray, flagged: set[str], doc: Document
    ) -> Finding | None:
        label_ids = self.cfg.label_ids
        weights = self.taxonomy.severity_weights
        shown: dict[str, tuple[str, list[str], float]] = {}
        for label in flagged:
            severity, notes = effective_severity(self.cfg, self.taxonomy, label, flagged)
            if severity in SHOWN_SEVERITIES:
                shown[label] = (severity, notes, float(scores[label_ids.index(label)]))
        if not shown:
            return None
        primary = max(shown, key=lambda lab: (weights.get(shown[lab][0], 0.0), shown[lab][2]))
        ordered = [
            primary,
            *sorted((lab for lab in shown if lab != primary), key=lambda lab: -shown[lab][2]),
        ]
        info = self.taxonomy.labels[primary]
        finding = Finding(
            module=self.cfg.module,
            unit_type=self.cfg.unit_type,
            unit_index=index,
            text=span.text,
            char_start=span.start,
            char_end=span.end,
            page=doc.page_of(span.start),
            labels=[
                LabelScore(
                    label=lab,
                    name=self.taxonomy.labels[lab].name,
                    score=round(shown[lab][2], 4),
                    severity=shown[lab][0],
                )
                for lab in ordered
            ],
            risk_group=info.risk_group,
            risk_group_name=self.taxonomy.risk_groups[info.risk_group].name,
            severity=shown[primary][0],
            explanation=template_explanation(ordered, notes=shown[primary][1]),
        )
        finding._notes = shown[primary][1]
        return finding

    def _explain_with_llm(self, findings: list[Finding]) -> None:
        """Replace template explanations with LLM ones where the LLM gives a valid answer."""
        if self.explainer is None or not self.explainer.enabled or not findings:
            return
        weights = self.taxonomy.severity_weights
        order = sorted(
            range(len(findings)), key=lambda i: (-weights.get(findings[i].severity, 0.0), i)
        )
        clauses = []
        for i in order:
            info = self.taxonomy.labels[findings[i].labels[0].label]
            clauses.append(Clause(findings[i].text, info.name, info.definition))
        for i, text in zip(order, self.explainer.explain(clauses), strict=True):
            if text:
                finding = findings[i]
                finding.explanation = compose_explanation(
                    text, [lab.label for lab in finding.labels], finding._notes
                )
                finding.explanation_source = "llm"

    def _merge_adjacent(self, findings: list[Finding], doc: Document) -> list[Finding]:
        """Join consecutive units flagged with the same primary label into one finding."""
        merged: list[Finding] = []
        for finding in findings:
            last = merged[-1] if merged else None
            if (
                last is not None
                and finding.unit_index == last.unit_index + last.n_units
                and finding.labels[0].label == last.labels[0].label
            ):
                scores = {lab.label: lab for lab in last.labels}
                for lab in finding.labels:
                    if lab.label not in scores or lab.score > scores[lab.label].score:
                        scores[lab.label] = lab
                labels = [scores[last.labels[0].label]] + sorted(
                    (lab for key, lab in scores.items() if key != last.labels[0].label),
                    key=lambda lab: -lab.score,
                )
                merged[-1] = last.model_copy(
                    update={
                        "text": doc.text[last.char_start : finding.char_end],
                        "char_end": finding.char_end,
                        "labels": labels,
                        "n_units": last.n_units + 1,
                    }
                )
            else:
                merged.append(finding)
        return merged

    def analyze(self, doc: Document) -> AnalysisResult:
        started = time.perf_counter()
        spans = self.segment(doc)
        units = [
            (i, s) for i, s in enumerate(spans) if is_classifiable(s.text, self.cfg.min_unit_words)
        ]
        segmented = time.perf_counter()
        probs = self.model.predict_proba([self.model_text(s.text) for _, s in units])
        predicted = time.perf_counter()

        label_ids = self.cfg.label_ids
        present: set[str] = set()
        findings = []
        flag_counts: Counter[str] = Counter()
        borderline = 0
        for (i, span), row in zip(units, probs, strict=True):
            hits = np.flatnonzero(row >= self.thresholds)
            flagged = {label_ids[j] for j in hits}
            flag_counts.update(flagged)
            borderline += int((row[hits] < self.thresholds[hits] + BORDERLINE_MARGIN).sum())
            present |= flagged
            finding = self._finding(i, span, row, flagged, doc)
            if finding is not None:
                findings.append(finding)
        if self.cfg.merge_adjacent:
            findings = self._merge_adjacent(findings, doc)
        self._explain_with_llm(findings)
        explained = time.perf_counter()

        found, missing, extra_risk = statements(self.cfg, self.taxonomy, present)
        items = [(lab.label, lab.severity, lab.score) for f in findings for lab in f.labels]
        score = score_from_raw(raw_risk(items, extra=extra_risk), self.risk_k)
        warnings = list(doc.warnings)
        if not present:
            warnings.append(
                f"No clauses typical of a {self.cfg.display_name} document were found. If this is "
                "a different kind of document, choose another document type."
            )
        return AnalysisResult(
            document_type=self.cfg.module,
            document_type_name=self.cfg.display_name,
            model=ModelInfo(
                run_id=self.meta["run_id"],
                model=self.meta["model"],
                trained_at=self.meta.get("created_at"),
            ),
            n_units=len(spans),
            n_words=doc.n_words,
            n_pages=doc.n_pages,
            risk_score=score,
            risk_band=risk_band(score),
            severity_counts=dict(Counter(f.severity for f in findings)),
            findings=findings,
            protections_found=found,
            protections_missing=missing,
            warnings=warnings,
            timings_ms={
                "segmentation": round(1000 * (segmented - started), 1),
                "classification": round(1000 * (predicted - segmented), 1),
                "findings_and_explanations": round(1000 * (explained - predicted), 1),
                "total": round(1000 * (time.perf_counter() - started), 1),
            },
            text=doc.text,
            diagnostics={
                "classified_units": len(units),
                "borderline_flags": borderline,
                "flags": dict(flag_counts),
            },
        )
