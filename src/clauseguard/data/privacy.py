"""OPP-115 → privacy unit table (one row per policy segment) + a sentence-level table.

OPP-115 policies are split into segments (roughly paragraphs); three law students annotated
each segment with data practices (a category plus attribute values, some tied to text spans).

* A segment gets a category when at least 2 of its 3 annotators gave it (majority vote).
* Sub-labels come from attribute values with the same majority rule, e.g. a Third Party
  Sharing practice whose Purpose is Advertising or Marketing.
* Span offsets index into the segment's raw HTML; ``strip_html`` + ``clean_text`` keep an
  offset map so spans land on the cleaned text. Spans listed in the corpus errata, or whose
  offsets don't match their ``selectedText``, are located by searching for the text instead.
* The sentence table gives a sentence a label when it overlaps a span of a practice with that
  label in a segment where the label won the vote. It is not used for training the paragraph
  model; it exists for finer highlighting and for combining modules later.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.splits import load_or_create_split
from clauseguard.data.units import COLUMNS, label_matrix, validate_units
from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.segment import is_classifiable, split_sentences
from clauseguard.paths import data_cards_dir, processed_dir, raw_dir
from clauseguard.textmap import strip_html

MAJORITY = 2
FIRST_PARTY = "First Party Collection/Use"
THIRD_PARTY = "Third Party Sharing/Collection"
NOT_PERSONAL_NOTICE = {
    "General notice in privacy policy",
    "General notice on website",
    "No notification",
    "Unspecified",
}


def _value(attrs: dict, name: str) -> str | None:
    return (attrs.get(name) or {}).get("value")


Rule = Callable[[str, dict], bool]
SUB_LABEL_RULES: dict[str, Rule] = {
    "privacy:sharing_ads_marketing": lambda cat, a: (
        cat == THIRD_PARTY and _value(a, "Purpose") in {"Advertising", "Marketing"}
    ),
    "privacy:negated_practice": lambda cat, a: (
        cat in {FIRST_PARTY, THIRD_PARTY} and _value(a, "Does/Does Not") == "Does Not"
    ),
    "privacy:change_without_personal_notice": lambda cat, a: (
        cat == "Policy Change" and _value(a, "Notification Type") in NOT_PERSONAL_NOTICE
    ),
}
# Recorded in unit metadata only: too rare to train (35 segments in the whole corpus).
RECORDED_RULES: dict[str, Rule] = {
    "retention_open_ended": lambda cat, a: (
        cat == "Data Retention" and _value(a, "Retention Period") in {"Indefinitely", "Unspecified"}
    ),
}


@dataclass
class Practice:
    annotator: str
    segment: int
    labels: set[str]
    recorded: set[str]
    spans: list[tuple[int, int]] = field(default_factory=list)  # raw-HTML offsets


def opp_root() -> Path:
    return raw_dir("opp115") / "OPP-115"


def read_errata(root: Path) -> set[tuple[str, str]]:
    """(annotation id, attribute name) pairs with known-bad span offsets."""
    bad = set()
    for path in (root / "documentation" / "errant_span_indexes").glob("*.csv"):
        for row in csv.reader(path.open(encoding="utf-8")):
            if len(row) >= 2:
                bad.add((row[0].strip(), row[1].split("_", 1)[-1].strip()))
    return bad


def _loose_search(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find ``needle`` in ``haystack`` treating any run of whitespace as equal."""
    words = needle.split()
    if not words:
        return None
    m = re.search(r"\s+".join(map(re.escape, words)), haystack)
    return m.span() if m else None


def read_practices(
    path: Path,
    segments: list[str],
    category_labels: dict[str, str],
    errata: set[tuple[str, str]],
    stats: Counter,
) -> list[Practice]:
    stripped = [strip_html(html) for html in segments]
    practices = []
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            annotation_id, _, annotator, _, segment_id, category, attrs_json = row[:7]
            if category not in category_labels:
                raise ValueError(f"{path.name}: unknown OPP-115 category {category!r}")
            attrs = json.loads(attrs_json)
            practice = Practice(
                annotator=annotator,
                segment=int(segment_id),
                labels={category_labels[category]}
                | {label for label, rule in SUB_LABEL_RULES.items() if rule(category, attrs)},
                recorded={name for name, rule in RECORDED_RULES.items() if rule(category, attrs)},
            )
            segment = segments[practice.segment]
            for name, attr in attrs.items():
                text = attr.get("selectedText")
                start = attr.get("startIndexInSegment", -1)
                end = attr.get("endIndexInSegment", -1)
                if not text or text == "null" or start is None or start < 0 or end <= start:
                    stats["spans_absent"] += 1
                    continue
                if (annotation_id, name) not in errata and segment[start:end] == text:
                    stats["spans_exact"] += 1
                    practice.spans.append((start, end))
                elif (found := segment.find(text)) >= 0:
                    stats["spans_found_by_search"] += 1
                    practice.spans.append((found, found + len(text)))
                elif match := _loose_search(text, stripped[practice.segment].text):
                    # whitespace in the selection differs from the page (tags, line breaks)
                    stats["spans_found_by_search"] += 1
                    practice.spans.append(stripped[practice.segment].to_source_span(*match))
                else:
                    stats["spans_unmapped"] += 1
            practices.append(practice)
    stats["practices"] += len(practices)
    return practices


def _policy_units(
    name: str,
    segments: list[str],
    practices: list[Practice],
    label_ids: list[str],
    stats: Counter,
) -> tuple[list[dict], list[dict]]:
    votes: dict[tuple[int, str], set[str]] = defaultdict(set)
    recorded_votes: dict[tuple[int, str], set[str]] = defaultdict(set)
    for p in practices:
        for label in p.labels:
            votes[(p.segment, label)].add(p.annotator)
        for name_ in p.recorded:
            recorded_votes[(p.segment, name_)].add(p.annotator)
    for (_, label), annotators in votes.items():
        stats[f"any:{label}"] += 1
        stats[f"majority:{label}"] += len(annotators) >= MAJORITY

    units, sentences = [], []
    offset = 0
    order = {label: i for i, label in enumerate(label_ids)}
    for sid, html in enumerate(segments):
        stripped = strip_html(html)
        cleaned = stripped.then(clean_text(stripped.text))
        text = cleaned.text
        won = sorted(
            (label for (s, label), a in votes.items() if s == sid and len(a) >= MAJORITY),
            key=order.__getitem__,
        )
        if not text.strip():
            stats["empty_segments"] += 1
            stats["labels_lost_on_empty_segments"] += len(won)
            continue
        recorded = {n for (s, n), a in recorded_votes.items() if s == sid and len(a) >= MAJORITY}
        units.append(
            {
                "doc_id": name,
                "unit_id": f"{name}:{sid}",
                "unit_type": "segment",
                "text": text,
                "char_start": offset,
                "char_end": offset + len(text),
                "labels": won,
                "exclude_from_training": False,
                "meta": json.dumps(
                    {
                        "segment_id": sid,
                        "votes": {label: len(a) for (s, label), a in votes.items() if s == sid},
                        **{n: n in recorded for n in RECORDED_RULES},
                    }
                ),
            }
        )

        label_spans = defaultdict(list)
        for p in practices:
            if p.segment != sid:
                continue
            for label in p.labels & set(won):
                for start, end in p.spans:
                    s, e = cleaned.from_source_span(start, end)
                    if e > s:
                        label_spans[label].append((s, e))
        for k, sentence in enumerate(split_sentences(text)):
            if not is_classifiable(sentence.text, 3):
                continue
            labels = [
                label
                for label in won
                if any(s < sentence.end and e > sentence.start for s, e in label_spans[label])
            ]
            sentences.append(
                {
                    "doc_id": name,
                    "unit_id": f"{name}:{sid}:{k}",
                    "unit_type": "sentence",
                    "text": sentence.text,
                    "char_start": offset + sentence.start,
                    "char_end": offset + sentence.end,
                    "labels": labels,
                    "exclude_from_training": False,
                    "meta": json.dumps({"segment_id": sid}),
                }
            )
        offset += len(text) + 2  # segments are joined with a blank line
    return units, sentences


def _frame(rows: list[dict], split_of: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["split"] = df["doc_id"].map(split_of)
    df["char_start"] = df["char_start"].astype("Int64")
    df["char_end"] = df["char_end"].astype("Int64")
    return df[COLUMNS]


def prepare_privacy() -> pd.DataFrame:
    cfg = load_module_config("privacy")
    root = opp_root()
    if not (root / "annotations").is_dir():
        raise FileNotFoundError(f"{root} not found; run `clauseguard data download privacy` first")
    category_labels = {s.source: s.id for s in cfg.labels if not s.source.startswith("rule:")}
    errata = read_errata(root)
    stats: Counter = Counter()
    units, sentences = [], []
    for path in sorted((root / "annotations").glob("*.csv")):
        name = path.stem
        segments = (
            (root / "sanitized_policies" / f"{name}.html").read_text(encoding="utf-8").split("|||")
        )
        practices = read_practices(path, segments, category_labels, errata, stats)
        stats["annotators_per_policy_" + str(len({p.annotator for p in practices}))] += 1
        policy_units, policy_sentences = _policy_units(
            name, segments, practices, cfg.label_ids, stats
        )
        units += policy_units
        sentences += policy_sentences
        stats["policies"] += 1

    label_sets: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        label_sets[unit["doc_id"]].update(unit["labels"])
    split_of = load_or_create_split(
        "privacy", dict(label_sets), cfg.data["split_ratios"], cfg.data["split_seed"]
    )
    df = _frame(units, split_of)
    sentence_df = _frame(sentences, split_of)
    validate_units(df, cfg.label_ids)
    validate_units(sentence_df, cfg.label_ids)

    out = processed_dir("privacy")
    out.mkdir(parents=True, exist_ok=True)
    sentence_df.to_parquet(out / "sentences.parquet", index=False)
    (out / "conversion_stats.json").write_text(json.dumps(dict(stats), indent=2, sort_keys=True))
    return df


def card_extras(df: pd.DataFrame) -> tuple[list[str], dict]:
    cfg = load_module_config("privacy")
    out = processed_dir("privacy")
    stats = json.loads((out / "conversion_stats.json").read_text())
    sentences = pd.read_parquet(out / "sentences.parquet")
    sentences["labels"] = sentences["labels"].map(list)

    md = ["## Conversion (OPP-115 → segments)", ""]
    md.append(
        f"{stats['policies']} policies, {stats['practices']:,} annotated practices; annotators per "
        f"policy: {stats.get('annotators_per_policy_3', 0)} policies with exactly 3. "
        f"{stats.get('empty_segments', 0)} segments were empty after removing HTML "
        f"({stats.get('labels_lost_on_empty_segments', 0)} majority labels lost)."
    )
    total = sum(stats.get(k, 0) for k in ("spans_exact", "spans_found_by_search", "spans_unmapped"))
    md.append(
        f"Span offsets: {stats.get('spans_exact', 0):,} exact, "
        f"{stats.get('spans_found_by_search', 0):,} found by searching for the selected text, "
        f"{stats.get('spans_unmapped', 0):,} unmapped "
        f"(of {total:,}); {stats.get('spans_absent', 0):,} attribute values have no span."
    )
    md += [
        "",
        "Agreement: segments where any annotator gave the label vs. the 2-of-3 majority.",
        "",
    ]
    md += md_table(
        ["label", "any annotator", "majority (used)"],
        [
            [f"`{label}`", stats.get(f"any:{label}", 0), stats.get(f"majority:{label}", 0)]
            for label in cfg.label_ids
        ],
    )

    y = label_matrix(sentences["labels"], cfg.label_ids)
    md += ["", "## Sentence-level table (derived, for highlighting and later combination)", ""]
    md.append(
        f"{len(sentences):,} sentences; {int(y.any(axis=1).sum()):,} carry at least one label. "
        "A sentence gets a label when it overlaps a span of that label's practices in a segment "
        "where the label won the vote."
    )
    md += [""] + md_table(
        ["label", "sentences"],
        [[f"`{lab}`", int(y[:, j].sum())] for j, lab in enumerate(cfg.label_ids)],
    )

    sample = pd.concat(
        [
            sentences[y.any(axis=1)].sample(35, random_state=0),
            sentences[~y.any(axis=1)].sample(15, random_state=0),
        ]
    )
    review = data_cards_dir() / "privacy_review_sample.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    sample.assign(labels=sample["labels"].map("; ".join), correct="", notes="")[
        ["unit_id", "text", "labels", "correct", "notes"]
    ].to_csv(review, index=False)
    md += [
        "",
        "## Manual check",
        "",
        f"50 random sentences (35 labelled, 15 unlabelled) are in `{review.name}`. Fill in the "
        "`correct` column (y/n) and record the error rate here.",
        "",
    ]
    return md, {"conversion": stats, "sentences": len(sentences)}
