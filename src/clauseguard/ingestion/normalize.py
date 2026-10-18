"""Make inference text look like each dataset's training text.

UNFAIR-ToS sentences were tokenised Penn-Treebank style by the dataset authors: lowercase,
punctuation separated by spaces, double quotes written as `` and '', contractions split
("are n't", "company 's"). Real documents look nothing like that, so model input is normalised
the same way. The function is idempotent on the training text itself (checked in tests).
"""

from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'(^|[\s(\[{])"'), r"\1 `` "),
    (re.compile(r'"'), " '' "),
    (re.compile(r"(\w)(n't)\b"), r"\1 \2"),
    (re.compile(r"(\w)('s|'re|'ve|'ll|'d|'m)\b"), r"\1 \2"),
    (re.compile(r"\s+-\s+"), " -- "),
    (re.compile(r"([()\[\]{}])"), r" \1 "),
    (re.compile(r"(?<=\S)([,;:?!])(?=\s|$)"), r" \1"),
    (re.compile(r"(?<=\S)\.(?=\s*$)"), " ."),
    (re.compile(r"\$(?=\d)"), "$ "),
    (re.compile(r"\s+"), " "),
]


def unfair_tos_style(text: str) -> str:
    text = text.lower()
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text.strip()
