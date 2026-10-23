"""Language check: ClauseGuard only supports English."""

from __future__ import annotations

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is random unless seeded
SAMPLE_CHARS = 5_000


def detect_language(text: str) -> tuple[str, float]:
    """Return (ISO code, probability) for the most likely language, or ("unknown", 0.0)."""
    sample = text[:SAMPLE_CHARS]
    try:
        best = detect_langs(sample)[0]
    except LangDetectException:
        return "unknown", 0.0
    return best.lang, float(best.prob)
