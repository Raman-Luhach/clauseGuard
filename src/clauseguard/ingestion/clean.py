"""Document text cleaning, shared by dataset conversion and inference.

``clean_text`` returns a ``TextWithMap`` so every position in the cleaned text can be traced back
to the raw input (and from there to a PDF page or a dataset span).
"""

from __future__ import annotations

import re
import unicodedata

from clauseguard.textmap import TextWithMap

_CHAR_MAP = {
    **dict.fromkeys("‘’‚‛′´`", "'"),
    **dict.fromkeys("“”„‟″«»", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
    **dict.fromkeys("            \t", " "),
    **dict.fromkeys("​‌‍⁠﻿­", ""),
    "…": "...",
    "\f": "\n\n",
    "\v": "\n",
    " ": "\n",
    " ": "\n\n",
}
_SPECIAL = re.compile(r"[^\x20-\x7e\n]")

_LIST_MARKER = re.compile(
    r"""^\s*(?:
        [•●○◦▪▫■□·‣⁃*]                      # bullet
      | -\s                                 # dash bullet
      | \(?\d{1,3}(?:\.\d{1,3})*[.)]\s?     # 1.  1)  (1)  4.2.
      | \(?[a-zA-Z][.)]\s                   # a.  a)  (a)
      | \([ivxlcIVXLC]{1,6}\)               # (iv)
      | (?:section|article|clause|schedule|exhibit|annex)\s+[\dIVXivx]
    )""",
    re.VERBOSE | re.IGNORECASE,
)
_ENDS_SENTENCE = re.compile(r"""[.!?:;]["')\]]*\s*$""")


def _normalise_char(m: re.Match[str]) -> str:
    ch = m.group()
    if ch in _CHAR_MAP:
        return _CHAR_MAP[ch]
    if unicodedata.category(ch).startswith("C"):  # control / format characters
        return ""
    return unicodedata.normalize("NFKC", ch)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or _ENDS_SENTENCE.search(stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    return len(stripped) < 40


def _unwrap(m: re.Match[str]) -> str:
    """Decide whether a single line break is a hard wrap (join) or structure (keep)."""
    text, start, end = m.string, m.start(), m.end()
    prev_line = text[text.rfind("\n", 0, start) + 1 : start]
    next_end = text.find("\n", end)
    next_line = text[end : next_end if next_end != -1 else len(text)]
    if not prev_line.strip() or not next_line.strip():
        return "\n"
    if _LIST_MARKER.match(next_line) or _ENDS_SENTENCE.search(prev_line):
        return "\n"
    if _looks_like_heading(prev_line):
        return "\n"
    first = next_line.lstrip()[0]
    if first.islower() or first.isdigit() or first in "(\"'$%&,":
        return " "
    if prev_line.rstrip().endswith((",", "-", "/", "&")) or len(prev_line) >= 50:
        return " "
    return "\n"


def clean_text(raw: str) -> TextWithMap:
    tm = TextWithMap.identity(raw)
    tm = tm.replace(r"\r\n?", "\n")
    tm = tm.replace(_SPECIAL, _normalise_char)
    tm = tm.replace(r"[ ]+(?=\n)", "")  # trailing spaces
    tm = tm.replace(r"(?<=\n)[ ]+", "")  # leading spaces
    tm = tm.replace(r"[ ]{2,}", " ")
    # "a.As of" / "(b)The" / "Article 1.Amendment": list markers glued to the next word
    tm = tm.replace(r"(?m)^(\(?(?:\d{1,3}|[a-zA-Z])[.)])(?=[A-Z][a-z])", lambda m: m.group(1) + " ")
    tm = tm.replace(r"(?<=\d\.)(?=[A-Z][a-z])", " ")
    # A hyphen at a line end is kept ("third-\nparty" -> "third-party"); only the break goes.
    tm = tm.replace(r"(?<=[A-Za-z]-)\n(?=[a-z])", "")
    tm = tm.replace(r"(?<!\n)\n(?!\n)", _unwrap)
    tm = tm.replace(r"\n[ ]*\n(?:[ ]*\n)+", "\n\n")
    return tm.strip()
