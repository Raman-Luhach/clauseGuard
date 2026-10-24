"""Entry point used by the CLI, API and UI: the caller always states the document type."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from clauseguard.config import available_modules, load_module_config
from clauseguard.errors import InputError
from clauseguard.explain.llm import Explainer, LLMSettings
from clauseguard.ingestion.loader import load_file, load_pdf, load_text
from clauseguard.modules.base import Analyzer
from clauseguard.schema import AnalysisResult

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


def analyze(
    document_type: str | None,
    *,
    text: str | None = None,
    path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    run_id: str | None = None,
) -> AnalysisResult:
    supported = [t["id"] for t in document_types()]
    if document_type not in supported:
        raise InputError(
            "unknown_document_type",
            f"Choose the document type first: one of {', '.join(supported)}.",
        )
    if sum(x is not None for x in (text, path, pdf_bytes)) != 1:
        raise ValueError("pass exactly one of text, path or pdf_bytes")
    analyzer = get_analyzer(document_type, run_id)
    content = text.encode() if text is not None else pdf_bytes or Path(path).read_bytes()
    key = hashlib.sha256(
        b"\x00".join(
            [
                document_type.encode(),
                analyzer.meta["run_id"].encode(),
                repr(get_explainer().settings).encode(),
                str(path).encode() if path is not None else b"",
                content,
            ]
        )
    ).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        cached = _cache[key].model_copy(deep=True)
        cached.diagnostics["cache_hit"] = True
        return cached
    if text is not None:
        doc = load_text(text)
    elif pdf_bytes is not None:
        doc = load_pdf(pdf_bytes)
    else:
        doc = load_file(path)
    result = analyzer.analyze(doc)
    _cache[key] = result.model_copy(deep=True)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def clear_cache() -> None:
    _cache.clear()
