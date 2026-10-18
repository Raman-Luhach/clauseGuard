"""Text extraction from digital (text-based) PDFs.

Pages are joined with a single line break so the cleaner can re-join sentences that continue
across a page break. Running headers, footers and page numbers are removed first, and the start
offset of every page is kept so findings can report their page.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from clauseguard.errors import InputError

MIN_CHARS_PER_PAGE = 40  # below this on average, treat the PDF as scanned / image-only
_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*(?:(?:of|/)\s*\d{1,4})?\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class PdfText:
    text: str
    page_starts: list[int]  # offset in ``text`` where each page begins
    n_pages: int
    empty_pages: int


def _edge_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _remove_running_lines(pages: list[list[str]]) -> list[list[str]]:
    """Drop page numbers and lines repeated at the top/bottom of most pages."""
    edge = 2
    counts: Counter[str] = Counter()
    for lines in pages:
        keys = {_edge_key(line) for line in lines[:edge] + lines[-edge:] if line.strip()}
        counts.update(keys)
    min_repeats = max(3, len(pages) // 2)
    repeated = {k for k, n in counts.items() if n >= min_repeats} if len(pages) >= 3 else set()

    cleaned = []
    for lines in pages:
        n = len(lines)
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < edge or i >= n - edge
            if at_edge and (_PAGE_NUMBER.match(line) or _edge_key(line) in repeated):
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned


def extract_pdf(source: str | Path | bytes) -> PdfText:
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        with pdfplumber.open(stream) as pdf:
            pages = [(page.dedupe_chars().extract_text() or "") for page in pdf.pages]
    except Exception as exc:  # pdfminer raises many unrelated exception types
        raise InputError(
            "unreadable_pdf", "This PDF could not be read. It may be damaged or password-protected."
        ) from exc
    if not pages:
        raise InputError("unreadable_pdf", "This PDF has no pages.")

    page_lines = _remove_running_lines([p.split("\n") for p in pages])
    page_texts = ["\n".join(lines).strip() for lines in page_lines]
    total_chars = sum(len(t) for t in page_texts)
    if total_chars / len(page_texts) < MIN_CHARS_PER_PAGE:
        raise InputError(
            "scanned_pdf",
            "This PDF has little or no selectable text, so it is probably a scan. ClauseGuard "
            "needs a text-based PDF; you can also paste the text instead.",
        )

    parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    for text in page_texts:
        page_starts.append(offset)
        parts.append(text)
        offset += len(text) + 1  # +1 for the joining "\n"
    return PdfText(
        text="\n".join(parts),
        page_starts=page_starts,
        n_pages=len(page_texts),
        empty_pages=sum(1 for t in page_texts if not t),
    )
