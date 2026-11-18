"""Entry point used by the CLI, API, UI and browser extension.

The caller states the document type, or passes ``"auto"`` to let the document-type router
choose it. The router only decides when it is confident; otherwise the caller is asked to
choose. When the caller did choose, the router's view is still attached to the result, and a
warning is added if the document looks like another type.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from clauseguard.config import available_modules, load_module_config
from clauseguard.errors import InputError, ModelNotAvailableError
from clauseguard.explain.llm import Explainer, LLMSettings
from clauseguard.ingestion.loader import Document, load_file, load_pdf, load_text
from clauseguard.modules.base import Analyzer
from clauseguard.schema import AnalysisResult, CombinedResult

AUTO = "auto"

CACHE_SIZE = 64  # recent results, keyed by content + document type + model run
_cache: OrderedDict[str, AnalysisResult] = OrderedDict()


def document_types() -> list[dict[str, str]]:
    return [
        {
            "id": m,
            "name": load_module_config(m).display_name,
            "description": load_module_config(m).description,
        }
        for m in available_modules()
    ]


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    """Shared by all analyzers; disabled unless CLAUSEGUARD_LLM=anthropic."""
    return Explainer(LLMSettings.from_env())


@lru_cache(maxsize=8)
def get_analyzer(document_type: str, run_id: str | None = None) -> Analyzer:
    return Analyzer(document_type, run_id, explainer=get_explainer())


def detect(doc: Document):
    """The router's detection for a document, or None when no router is installed."""
    from clauseguard.router import get_router

    try:
        router = get_router()
    except ModelNotAvailableError:
        return None
    return router.detect(doc.text)


def router_ready() -> bool:
    from clauseguard.router import get_router

    try:
        get_router()
    except ModelNotAvailableError:
        return False
    return True


def load_document(
    *, text: str | None = None, path: str | Path | None = None, pdf_bytes: bytes | None = None
) -> Document:
    if sum(x is not None for x in (text, path, pdf_bytes)) != 1:
        raise ValueError("pass exactly one of text, path or pdf_bytes")
    if text is not None:
        return load_text(text)
    if pdf_bytes is not None:
        return load_pdf(pdf_bytes)
    return load_file(path)


def _type_name(doc_type: str) -> str:
    return load_module_config(doc_type).display_name


def _choose_type(document_type: str | None, detection, supported: list[str]) -> str:
    if document_type not in (None, AUTO):
        if document_type not in supported:
            raise InputError(
                "unknown_document_type",
                f"Choose the document type first: one of {', '.join(supported)}.",
            )
        return document_type
    if detection is None:
        raise InputError(
            "unknown_document_type",
            f"Choose the document type: one of {', '.join(supported)} "
            "(automatic detection is not installed).",
        )
    details = {"detection": detection.as_dict()}
    if detection.unrelated:
        raise InputError(
            "unrelated_document",
            "This doesn't look like a terms of service, a privacy policy or a contract. If it "
            "is one, choose the document type yourself.",
            details,
        )
    if not detection.confident or detection.document_type not in supported:
        guess = max(
            (t for t in detection.probabilities if t in supported),
            key=detection.probabilities.get,
        )
        raise InputError(
            "uncertain_document_type",
            f"ClauseGuard isn't sure what kind of document this is (maybe a {_type_name(guess)}"
            f", {detection.probabilities[guess]:.0%}). Choose the document type.",
            details,
        )
    return detection.document_type


def _mismatch_warning(chosen: str, detection) -> str | None:
    if detection is None or not detection.confident or detection.document_type == chosen:
        return None
    other = detection.document_type
    return (
        f"This document looks more like a {_type_name(other)} "
        f"({detection.probabilities[other]:.0%}) than a {_type_name(chosen)}. If so, choose "
        f"{_type_name(other)} and analyse it again."
    )


def analyze(
    document_type: str | None,
    *,
    text: str | None = None,
    path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    run_id: str | None = None,
) -> AnalysisResult:
    """Analyse one document. ``document_type`` is a module id, or ``"auto"`` / None to let the
    router choose (it raises InputError when the router isn't confident)."""
    supported = [t["id"] for t in document_types()]
    if document_type not in (None, AUTO, *supported):
        raise InputError(
            "unknown_document_type",
            f"Choose the document type first: one of {', '.join(supported)}.",
        )
    if sum(x is not None for x in (text, path, pdf_bytes)) != 1:
        raise ValueError("pass exactly one of text, path or pdf_bytes")
    content = text.encode() if text is not None else pdf_bytes or Path(path).read_bytes()
    doc = load_document(text=text, path=path, pdf_bytes=pdf_bytes)
    detection = detect(doc)
    chosen = _choose_type(document_type, detection, supported)
    analyzer = get_analyzer(chosen, run_id)
    key = hashlib.sha256(
        b"\x00".join(
            [
                chosen.encode(),
                analyzer.meta["run_id"].encode(),
                repr(get_explainer().settings).encode(),
                str(path).encode() if path is not None else b"",
                content,
            ]
        )
    ).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        result = _cache[key].model_copy(deep=True)
        result.diagnostics["cache_hit"] = True
    else:
        result = analyzer.analyze(doc)
        _cache[key] = result.model_copy(deep=True)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    result.detection = detection.as_dict() if detection is not None else None
    result.type_source = "user" if document_type not in (None, AUTO) else "router"
    warning = _mismatch_warning(chosen, detection) if result.type_source == "user" else None
    if warning:
        result.warnings.insert(0, warning)
    return result


def analyze_combined(
    document_types_: list[str] | None = None,
    *,
    text: str | None = None,
    path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
) -> CombinedResult:
    """Run several analyzers on one document (default: every available one). Useful for pages
    that combine terms of service and a privacy policy."""
    supported = [t["id"] for t in document_types()]
    chosen = document_types_ or supported
    unknown = [t for t in chosen if t not in supported]
    if unknown:
        raise InputError(
            "unknown_document_type",
            f"Unknown document types {unknown}; choose from {', '.join(supported)}.",
        )
    results = [analyze(t, text=text, path=path, pdf_bytes=pdf_bytes) for t in chosen]
    return CombinedResult(results=results, detection=results[0].detection if results else None)


def clear_cache() -> None:
    _cache.clear()
