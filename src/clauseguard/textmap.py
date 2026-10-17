"""Text transformations that remember where every character came from.

Cleaning a document (stripping HTML, unwrapping lines, normalising quotes) changes character
positions. Dataset labels (OPP-115 spans, CUAD spans) and report offsets must survive those
changes, so every transformation here returns a new ``TextWithMap`` whose ``origin[i]`` is the
index in the original source of output character ``i``. Origins are always non-decreasing, which
lets us map spans in both directions with binary search.
"""

from __future__ import annotations

import html
import re
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass

Replacement = str | Callable[[re.Match[str]], str]

_TAG = re.compile(r"<[^>]*>")
_BLOCK_TAG = re.compile(
    r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|header|footer|blockquote)\b[^>]*>",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")


@dataclass(frozen=True, slots=True)
class TextWithMap:
    text: str
    origin: tuple[int, ...]
    source_len: int

    def __post_init__(self) -> None:
        if len(self.text) != len(self.origin):
            raise ValueError("text and origin must have the same length")

    @classmethod
    def identity(cls, source: str) -> TextWithMap:
        return cls(source, tuple(range(len(source))), len(source))

    def __len__(self) -> int:
        return len(self.text)

    def replace(self, pattern: str | re.Pattern[str], repl: Replacement) -> TextWithMap:
        """Regex substitution that keeps the origin map.

        Replacement characters map onto the matched characters in order; extra characters
        take the origin of the last matched character. Pure insertions (empty matches) take
        the origin of the preceding character, so they never leak into the following word.
        """
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        text, origin = self.text, self.origin
        out_text: list[str] = []
        out_origin: list[int] = []
        pos = 0
        for m in regex.finditer(text):
            start, end = m.span()
            out_text.append(text[pos:start])
            out_origin.extend(origin[pos:start])
            new = repl(m) if callable(repl) else m.expand(repl)
            out_text.append(new)
            if end > start:
                out_origin.extend(origin[start + min(i, end - start - 1)] for i in range(len(new)))
            else:
                anchor = origin[start - 1] if start > 0 else (origin[0] if origin else 0)
                out_origin.extend([anchor] * len(new))
            pos = end
        out_text.append(text[pos:])
        out_origin.extend(origin[pos:])
        return TextWithMap("".join(out_text), tuple(out_origin), self.source_len)

    def then(self, later: TextWithMap) -> TextWithMap:
        """Compose: ``later`` was computed from ``self.text``; map it back to our source."""
        last = len(self.origin) - 1
        origin = tuple(self.origin[min(o, last)] if last >= 0 else 0 for o in later.origin)
        return TextWithMap(later.text, origin, self.source_len)

    def strip(self) -> TextWithMap:
        start = len(self.text) - len(self.text.lstrip())
        end = len(self.text.rstrip())
        return TextWithMap(self.text[start:end], self.origin[start:end], self.source_len)

    def to_source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span of this text to a span of the original source."""
        if not 0 <= start <= end <= len(self.text):
            raise IndexError(f"span ({start}, {end}) outside text of length {len(self.text)}")
        if start == end:
            pos = self.origin[start] if start < len(self.origin) else self.source_len
            return pos, pos
        return self.origin[start], self.origin[end - 1] + 1

    def from_source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span of the original source onto this text.

        Source characters that were removed (tags, collapsed whitespace) shrink the span to
        the surviving characters inside it; surrounding whitespace is trimmed.
        """
        lo, hi = bisect_left(self.origin, start), bisect_left(self.origin, end)
        while lo < hi and self.text[lo].isspace():
            lo += 1
        while hi > lo and self.text[hi - 1].isspace():
            hi -= 1
        return lo, hi


def strip_html(source: str) -> TextWithMap:
    """Remove HTML tags and decode entities, following how browsers render text: line breaks
    in the source are just spaces, and only block-level tags start a new line."""
    tm = TextWithMap.identity(source)
    tm = tm.replace(r"[ \t]*[\r\n]+[ \t]*", " ")
    tm = tm.replace(_BLOCK_TAG, lambda m: "\n")
    tm = tm.replace(_TAG, "")
    return tm.replace(_ENTITY, lambda m: html.unescape(m.group()))
