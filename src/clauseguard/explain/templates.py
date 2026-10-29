"""Template explanations: always available, used when no LLM is configured or it fails."""

from __future__ import annotations

from clauseguard.config import load_taxonomy


def compose_explanation(primary: str, labels: list[str], notes: list[str] | None = None) -> str:
    """``primary`` explains the first label; modifier ``notes`` and the other labels follow."""
    taxonomy = load_taxonomy()
    text = " ".join([primary.strip(), *(notes or [])])
    others = [taxonomy.labels[label].name for label in labels[1:]]
    if others:
        text += f" This clause also concerns: {', '.join(others)}."
    return text


def template_explanation(labels: list[str], notes: list[str] | None = None) -> str:
    """Explain a clause flagged with ``labels`` (the first is the primary one). ``notes`` come
    from severity modifiers, e.g. "The sharing appears to be for advertising or marketing."."""
    return compose_explanation(load_taxonomy().labels[labels[0]].template, labels, notes)
