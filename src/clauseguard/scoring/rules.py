"""Severity modifiers and expected statements, configured per module.

Used by both the analyzer and the risk-score calibration, so they always agree.
"""

from __future__ import annotations

from clauseguard.config import ModuleConfig, Taxonomy

SEVERITY_LADDER = ("info", "low", "medium", "high")


def effective_severity(
    cfg: ModuleConfig, taxonomy: Taxonomy, label: str, flagged: set[str]
) -> tuple[str, list[str]]:
    """Severity of ``label`` given every label predicted for the same unit.

    Each configured modifier that is also predicted moves the severity one step up or down
    the ladder info < low < medium < high, and contributes its template as an explanation
    note. Protective and hidden labels are never modified.
    """
    severity = taxonomy.labels[label].severity
    if severity not in SEVERITY_LADDER:
        return severity, []
    level = SEVERITY_LADDER.index(severity)
    notes = []
    for modifier in cfg.modifiers:
        if label in modifier.applies_to and modifier.label in flagged:
            level = min(max(level + modifier.change, 0), len(SEVERITY_LADDER) - 1)
            notes.append(taxonomy.labels[modifier.label].template.strip())
    return SEVERITY_LADDER[level], notes


def statements(
    cfg: ModuleConfig, taxonomy: Taxonomy, present: set[str]
) -> tuple[list[str], list[str], float]:
    """(found names, missing messages, extra raw risk) for the module's expected statements."""
    found = [taxonomy.labels[label].name for label in cfg.expected_statements if label in present]
    missing = [
        taxonomy.labels[label].missing_message
        or f"No clear statement about {taxonomy.labels[label].name.lower()}."
        for label in cfg.expected_statements
        if label not in present
    ]
    return found, missing, cfg.missing_statement_weight * len(missing)
