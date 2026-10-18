"""Turn an uploaded PDF or pasted text into a cleaned, validated ``Document``."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from clauseguard.errors import InputError
from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.language import detect_language
from clauseguard.ingestion.pdf import extract_pdf
from clauseguard.textmap import TextWithMap

MIN_WORDS = 50
MAX_WORDS = 150_000
MAX_PDF_BYTES = 20 * 1024 * 1024
NON_ENGLISH_CONFIDENCE = 0.80


@dataclass
class Document:
    raw: str
    clean: TextWithMap
    source: Literal["pdf", "text"]
    page_starts: list[int] | None = None  # raw offsets, PDFs only
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.clean.text

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def n_pages(self) -> int | None:
        return len(self.page_starts) if self.page_starts is not None else None

    def page_of(self, start: int) -> int | None:
        """1-based page of a character offset in the cleaned text."""
        if self.page_starts is None or not 0 <= start < len(self.clean.origin):
            return None
        return bisect_right(self.page_starts, self.clean.origin[start])


def _validate(doc: Document) -> Document:
    n = doc.n_words
    if n == 0:
        raise InputError("empty_input", "The document is empty.")
    if n < MIN_WORDS:
        raise InputError(
            "too_short",
            f"The document has only {n} words. Please provide the full document "
            f"(at least {MIN_WORDS} words).",
        )
    if n > MAX_WORDS:
        raise InputError(
            "too_long",
            f"The document has {n:,} words; the limit is {MAX_WORDS:,}. "
            "Please split it into parts.",
        )
    lang, prob = detect_language(doc.text)
    if lang != "en" and prob >= NON_ENGLISH_CONFIDENCE:
        raise InputError(
            "not_english",
            f"The document appears to be in a language other than English ({lang}). "
            "ClauseGuard currently supports English only.",
        )
    return doc


def load_text(text: str) -> Document:
    return _validate(Document(raw=text, clean=clean_text(text), source="text"))


def load_pdf(source: str | Path | bytes) -> Document:
    size = len(source) if isinstance(source, bytes) else Path(source).stat().st_size
    if size > MAX_PDF_BYTES:
        raise InputError(
            "file_too_large", f"The PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB."
        )
    pdf = extract_pdf(source)
    doc = Document(
        raw=pdf.text, clean=clean_text(pdf.text), source="pdf", page_starts=pdf.page_starts
    )
    if pdf.empty_pages:
        doc.warnings.append(
            f"{pdf.empty_pages} of {pdf.n_pages} pages had no selectable text and were skipped."
        )
    return _validate(doc)


def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text(path.read_text(encoding="utf-8", errors="replace"))
