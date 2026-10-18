"""Sentence segmentation for legal text, shared by dataset conversion and inference.

Lines in cleaned text are structural (``clean_text`` already joined hard-wrapped lines), so each
line is split separately with spaCy's statistical sentence splitter (``senter``, about 13x faster
than the full parser). Rules then fix the usual legal-text mistakes: abbreviations ("Inc.",
"No. 5", "e.g."), bare list markers ("(a)", "4.2"), tiny fragments and run-on clauses.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language

MAX_SPACY_CHARS = 200_000

_HARD_ABBREV = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|st|no|nos|sec|secs|art|arts|para|paras|cl|fig|ex|exh|sch|cf|viz|"
    r"e\.g|i\.e|vs|v|pp|approx|incl)\.$",
    re.IGNORECASE,
)
_SOFT_ABBREV = re.compile(
    r"\b(?:inc|ltd|co|corp|llc|l\.l\.c|llp|l\.p|plc|pty|etc|u\.s|u\.k|u\.s\.a|e\.u|jr|sr)\.$",
    re.IGNORECASE,
)
_MARKER_ONLY = re.compile(
    r"^(?:\(?[0-9]{1,3}(?:\.[0-9]{1,3})*[.)]?|\(?[a-zA-Z][.)]|\([ivxlcIVXLC]{1,6}\)|"
    r"(?:section|article|clause)\s+[0-9.]+)$",
    re.IGNORECASE,
)


_CLOSING_PUNCT = re.compile(r"""[.;:!?,]["')\]]*$""")


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    text: str


def looks_like_heading(text: str, max_words: int = 12) -> bool:
    """Short, unpunctuated, mostly capitalised units ("3. GitHub May Terminate").

    Headings repeat the topic of the clause below them, so classifying them would report the
    same clause twice.
    """
    words = text.split()
    if not words or len(words) > max_words or _CLOSING_PUNCT.search(text.rstrip()):
        return False
    alpha = [w for w in words if w[0].isalpha()]
    if not alpha:
        return True
    return sum(w[0].isupper() for w in alpha) / len(alpha) >= 0.6


@lru_cache(maxsize=1)
def _nlp() -> Language:
    nlp = spacy.load(
        "en_core_web_sm",
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    nlp.enable_pipe("senter")
    nlp.max_length = MAX_SPACY_CHARS + 1_000
    return nlp


def _lines(text: str) -> Iterator[tuple[int, int]]:
    """Non-empty lines, with very long ones cut at sentence-like boundaries for spaCy."""
    for m in re.finditer(r"[^\n]+", text):
        start, end = m.span()
        if not text[start:end].strip():
            continue
        while end - start > MAX_SPACY_CHARS:
            window = text[start : start + MAX_SPACY_CHARS]
            cut = max((c.end() for c in re.finditer(r"[.!?]\s+", window)), default=0)
            cut = start + (cut if cut > MAX_SPACY_CHARS // 2 else MAX_SPACY_CHARS)
            yield start, cut
            start = cut
        yield start, end


def _should_merge(prev: str, cur: str, doc_has_upper: bool) -> bool:
    prev, cur = prev.rstrip(), cur.lstrip()
    if not cur or cur[0] in ".,;:)]":
        return True
    if _MARKER_ONLY.match(prev) or _HARD_ABBREV.search(prev):
        return True
    continues = cur[0].islower() or cur[0].isdigit() or cur[0] == "("
    if _SOFT_ABBREV.search(prev) and continues:
        return True
    # In normally-cased text a sentence almost never starts in lowercase.
    return doc_has_upper and cur[0].islower()


def _merge(text: str, spans: list[tuple[int, int]], doc_has_upper: bool, min_chars: int):
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and _should_merge(text[slice(*merged[-1])], text[start:end], doc_has_upper):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    out: list[tuple[int, int]] = []
    for span in merged:
        if out and len(text[slice(*out[-1])].strip()) < min_chars:
            out[-1] = (out[-1][0], span[1])
        else:
            out.append(span)
    if len(out) > 1 and len(text[slice(*out[-1])].strip()) < min_chars:
        last = out.pop()
        out[-1] = (out[-1][0], last[1])
    return out


def _split_long(text: str, start: int, end: int, max_chars: int) -> Iterator[tuple[int, int]]:
    """Cut run-on units at '; ', then ', ', then any whitespace."""
    while end - start > max_chars:
        window = text[start : start + max_chars]
        cut = None
        for pattern in (r";\s", r",\s", r"\s"):
            cuts = [m.end() for m in re.finditer(pattern, window) if m.end() >= max_chars // 3]
            if cuts:
                cut = start + cuts[-1]
                break
        cut = cut or start + max_chars
        yield start, cut
        start = cut
    yield start, end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_sentences(text: str, *, min_chars: int = 20, max_chars: int = 1500) -> list[Span]:
    """Split cleaned text into sentence spans whose offsets index into ``text``."""
    lines = list(_lines(text))
    doc_has_upper = any(c.isupper() for c in text)
    sentences: list[Span] = []
    docs = _nlp().pipe((text[s:e] for s, e in lines), batch_size=64)
    for (line_start, _), doc in zip(lines, docs, strict=True):
        raw = [(line_start + s.start_char, line_start + s.end_char) for s in doc.sents]
        for start, end in _merge(text, raw, doc_has_upper, min_chars):
            for s, e in _split_long(text, start, end, max_chars):
                s, e = _trim(text, s, e)
                if e > s:
                    sentences.append(Span(s, e, text[s:e]))
    return sentences


def is_classifiable(text: str, min_words: int) -> bool:
    """Units worth classifying: long enough to be a clause and not a heading."""
    return len(text.split()) >= min_words and not looks_like_heading(text)


def _words(text: str, spans: list[tuple[int, int]]) -> int:
    return sum(len(text[s:e].split()) for s, e in spans)


def split_paragraphs(text: str, *, min_words: int = 12, max_words: int = 180) -> list[Span]:
    """Split cleaned text into paragraph-sized units comparable to OPP-115 segments.

    A new paragraph starts at a blank line or a heading; a heading stays attached to the text
    that follows it. Paragraphs shorter than ``min_words`` are merged into the next one, and
    longer than ``max_words`` are cut at sentence boundaries. This also works for PDF text,
    where paragraphs are rarely separated by blank lines.
    """
    groups: list[list[tuple[int, int]]] = []
    prev_end: int | None = None
    for m in re.finditer(r"[^\n]+", text):
        start, end = m.span()
        if not text[start:end].strip():
            continue
        blank_before = prev_end is not None and text.count("\n", prev_end, start) >= 2
        heading = looks_like_heading(text[start:end])
        follows_heading = bool(groups) and looks_like_heading(text[slice(*groups[-1][-1])])
        if (
            not groups
            or blank_before
            or (heading and not follows_heading)
            or _words(text, groups[-1]) >= max_words
        ):
            groups.append([(start, end)])
        else:
            groups[-1].append((start, end))
        prev_end = end

    spans = [(g[0][0], g[-1][1]) for g in groups]
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and len(text[slice(*merged[-1])].split()) < min_words:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    if len(merged) > 1 and len(text[slice(*merged[-1])].split()) < min_words:
        last = merged.pop()
        merged[-1] = (merged[-1][0], last[1])

    paragraphs: list[Span] = []
    for start, end in merged:
        if len(text[start:end].split()) <= max_words:
            paragraphs.append(Span(start, end, text[start:end]))
            continue
        chunk_start, chunk_end, chunk_words = None, start, 0
        for sentence in split_sentences(text[start:end]):
            n = len(sentence.text.split())
            if chunk_start is not None and chunk_words + n > max_words:
                paragraphs.append(Span(chunk_start, chunk_end, text[chunk_start:chunk_end]))
                chunk_start, chunk_words = None, 0
            if chunk_start is None:
                chunk_start = start + sentence.start
            chunk_end = start + sentence.end
            chunk_words += n
        if chunk_start is not None:
            paragraphs.append(Span(chunk_start, chunk_end, text[chunk_start:chunk_end]))
    return paragraphs
