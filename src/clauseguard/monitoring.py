"""Request monitoring: metadata only, never document text.

Every analysis and every feedback click is appended to ``logs/events.jsonl`` (override with
``CLAUSEGUARD_LOG_DIR``). ``summarize`` compares recent traffic with the baseline each training
run stores (how often each label fires per unit, and how many flags sit just above their
threshold): a label firing far more or less often than in validation, or many borderline flags,
suggests the incoming documents differ from the training data.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from clauseguard.paths import root

BORDERLINE_MARGIN = 0.1  # a flag scoring less than this above its threshold is borderline
DRIFT_RATIO = 2.0  # a label firing 2x more, or 2x less, often than in validation
MIN_UNITS_FOR_DRIFT = 200
_lock = threading.Lock()


def log_dir() -> Path:
    return Path(os.environ.get("CLAUSEGUARD_LOG_DIR", root() / "logs"))


def _append(event: dict) -> None:
    event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    path = log_dir() / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def log_analysis(result, *, source: str, latency_ms: float) -> None:
    """``result`` is an AnalysisResult; only counts and scores are recorded."""
    flags = Counter(result.diagnostics.get("flags", {}))
    _append(
        {
            "event": "analysis",
            "status": "ok",
            "document_type": result.document_type,
            "model_run": result.model.run_id,
            "source": source,
            "n_words": result.n_words,
            "n_units": result.n_units,
            "classified_units": int(result.diagnostics.get("classified_units", 0)),
            "borderline_flags": int(result.diagnostics.get("borderline_flags", 0)),
            "n_flags": sum(flags.values()),
            "flags": dict(flags),
            "severity_counts": result.severity_counts,
            "risk_score": result.risk_score,
            "explanations": dict(Counter(f.explanation_source for f in result.findings)),
            "latency_ms": round(latency_ms, 1),
        }
    )


def log_error(document_type: str | None, code: str, *, source: str, latency_ms: float) -> None:
    _append(
        {
            "event": "analysis",
            "status": "error",
            "error_code": code,
            "document_type": document_type,
            "source": source,
            "latency_ms": round(latency_ms, 1),
        }
    )


def log_feedback(document_type: str, model_run: str, label: str, agree: bool) -> None:
    _append(
        {
            "event": "feedback",
            "document_type": document_type,
            "model_run": model_run,
            "label": label,
            "agree": bool(agree),
        }
    )


def read_events(since: timedelta | None = None) -> list[dict]:
    path = log_dir() / "events.jsonl"
    if not path.exists():
        return []
    cutoff = datetime.now(UTC) - since if since else None
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if cutoff is None or datetime.fromisoformat(event["ts"]) >= cutoff:
            events.append(event)
    return events


def baseline_from_validation(p: np.ndarray, thresholds: list[float], label_ids: list[str]) -> dict:
    """What normal looks like, from a run's validation predictions."""
    t = np.asarray(thresholds)
    flagged = p >= t
    borderline = flagged & (p < t + BORDERLINE_MARGIN)
    n = max(len(p), 1)
    return {
        "units": int(len(p)),
        "flags_per_unit": {
            label: float(flagged[:, j].sum() / n) for j, label in enumerate(label_ids)
        },
        "borderline_share": float(borderline.sum() / max(flagged.sum(), 1)),
    }


def summarize(since: timedelta | None = None, baselines: dict[str, dict] | None = None) -> dict:
    """Per document type: volume, errors, latency, flag rates and drift against ``baselines``
    (document type -> the baseline of the run that served it)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    feedback: dict[str, list[bool]] = defaultdict(list)
    for event in read_events(since):
        if event["event"] == "analysis":
            by_type[event.get("document_type") or "unknown"].append(event)
        elif event["event"] == "feedback":
            feedback[event["document_type"]].append(event["agree"])

    summary = {}
    for doc_type, events in sorted(by_type.items()):
        ok = [e for e in events if e["status"] == "ok"]
        latencies = [e["latency_ms"] for e in ok]
        units = sum(e["classified_units"] for e in ok)
        flags: Counter[str] = Counter()
        for e in ok:
            flags.update(e["flags"])
        n_flags = sum(e["n_flags"] for e in ok)
        entry = {
            "requests": len(events),
            "errors": dict(Counter(e["error_code"] for e in events if e["status"] == "error")),
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "median_words": float(np.median([e["n_words"] for e in ok])) if ok else None,
            "classified_units": units,
            "borderline_share": sum(e["borderline_flags"] for e in ok) / n_flags
            if n_flags
            else None,
            "feedback_agree_rate": float(np.mean(feedback[doc_type]))
            if feedback[doc_type]
            else None,
            "feedback_count": len(feedback[doc_type]),
            "drift": [],
        }
        baseline = (baselines or {}).get(doc_type)
        if baseline and units >= MIN_UNITS_FOR_DRIFT:
            for label, expected in baseline["flags_per_unit"].items():
                observed = flags.get(label, 0) / units
                if expected > 0 and not 1 / DRIFT_RATIO <= observed / expected <= DRIFT_RATIO:
                    entry["drift"].append(
                        {
                            "label": label,
                            "expected_per_unit": expected,
                            "observed_per_unit": observed,
                        }
                    )
            if entry["borderline_share"] is not None and (
                entry["borderline_share"] > DRIFT_RATIO * max(baseline["borderline_share"], 0.05)
            ):
                entry["drift"].append(
                    {
                        "label": "borderline_share",
                        "expected": baseline["borderline_share"],
                        "observed": entry["borderline_share"],
                    }
                )
        summary[doc_type] = entry
    return summary
