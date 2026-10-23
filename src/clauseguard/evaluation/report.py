"""Markdown evaluation reports."""

from __future__ import annotations

import math

import pandas as pd

from clauseguard.data.card import md_table


def _f(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _prf_cell(m: dict) -> str:
    return f"{_f(m['precision'])} / {_f(m['recall'])} / {_f(m['f1'])}"


def metrics_markdown(title: str, metrics: dict, confusions: pd.DataFrame | None = None) -> str:
    md = [f"# {title}", "", f"Units evaluated: {metrics['n']}.", "", "## Summary", ""]
    rows = [
        ["micro precision / recall / F1", _prf_cell(metrics["micro"])],
        [f"macro F1 ({metrics['macro_labels']} labels present)", _f(metrics["macro_f1"])],
        ["macro PR-AUC", _f(metrics["macro_pr_auc"])],
        ["any risky clause: precision / recall / F1", _prf_cell(metrics["any_risk"])],
    ]
    if "lexglue" in metrics:
        rows.append(
            [
                "LexGLUE-style micro / macro F1 (with 'none' class)",
                f"{_f(metrics['lexglue']['micro_f1'])} / {_f(metrics['lexglue']['macro_f1'])}",
            ]
        )
    md += md_table(["metric", "value"], rows)
    md += ["", "## Per label", ""]
    md += md_table(
        ["label", "support", "predicted", "precision", "recall", "F1", "PR-AUC", "threshold"],
        [
            [
                f"`{label}`",
                m["support"],
                m["predicted"],
                _f(m["precision"]),
                _f(m["recall"]),
                _f(m["f1"]),
                _f(m["pr_auc"]),
                f"{m['threshold']:.3f}",
            ]
            for label, m in metrics["per_label"].items()
        ],
    )
    risk = metrics.get("risk_score")
    if risk and risk["documents"] >= 3:
        md += [
            "",
            "## Risk score ranking",
            "",
            f"Spearman correlation between documents' risk from predictions and from the gold "
            f"labels: {_f(risk['spearman'])} over {risk['documents']} documents (1 = same order).",
        ]
    documents = metrics.get("documents")
    if documents:
        md += [
            "",
            f"## Document level ({documents['n_documents']} documents)",
            "",
            "Does the document contain the label anywhere? Macro F1 "
            f"{_f(documents['macro_f1'])}, mean accuracy {_f(documents['mean_accuracy'])}.",
            "",
        ]
        md += md_table(
            ["label", "documents with it", "precision", "recall", "F1", "accuracy"],
            [
                [
                    f"`{label}`",
                    m["documents_with_label"],
                    _f(m["precision"]),
                    _f(m["recall"]),
                    _f(m["f1"]),
                    _f(m["accuracy"]),
                ]
                for label, m in documents["per_label"].items()
            ],
        )
    clauses = metrics.get("clauses")
    if clauses:
        md += [
            "",
            f"## Clause recall ({clauses['n_spans']} gold spans)",
            "",
            "Share of gold clauses flagged with the right label in at least one overlapping "
            f"unit. Micro {_f(clauses['micro_recall'])}, macro {_f(clauses['macro_recall'])}.",
            "",
        ]
        md += md_table(
            ["label", "gold spans", "recall"],
            [
                [f"`{label}`", m["spans"], _f(m["recall"])]
                for label, m in clauses["per_label"].items()
            ],
        )
    if confusions is not None and len(confusions):
        md += [
            "",
            "## Most common errors",
            "",
            "`(none)` as gold = false alarm; `(none)` as predicted = missed clause.",
            "",
        ]
        md += md_table(["predicted", "gold", "count"], confusions.head(15).values.tolist())
    return "\n".join(md) + "\n"
