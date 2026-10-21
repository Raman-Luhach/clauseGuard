"""CUAD v1 → contract unit table (one row per sentence) + the gold spans.

CUAD marks clauses as character spans in each contract (SQuAD format). Contracts are cleaned
with an offset map, split into sentences with the shared splitter, and each sentence gets a
category when it overlaps a span of that category by at least ``overlap_threshold`` times the
shorter of the two. That catches a short span inside a long sentence and a long span covering
several sentences, and ignores spans that spill a few characters into the next sentence.
Sentences with a smaller, non-zero overlap are ambiguous: kept for evaluation, excluded from
training. The official split is by contract (408 train / 102 test); validation contracts are
carved out of the training contracts.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.splits import load_or_create_split
from clauseguard.data.units import COLUMNS, label_matrix, validate_units
from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.segment import Span, is_classifiable, split_sentences
from clauseguard.paths import data_cards_dir, processed_dir, raw_dir


def cuad_root() -> Path:
    return raw_dir("cuad")


def _titles(path: Path) -> set[str]:
    return {doc["title"] for doc in json.loads(path.read_text(encoding="utf-8"))["data"]}


def label_sentences(
    sentences: list[Span], spans: list[tuple[str, int, int]], threshold: float
) -> tuple[list[set[str]], list[set[str]], list[bool]]:
    """Per sentence: (positive labels, ambiguous labels); per span: covered by a positive."""
    starts = [s.start for s in sentences]
    positive = [set() for _ in sentences]
    ambiguous = [set() for _ in sentences]
    covered = []
    for label, start, end in spans:
        hit = False
        i = max(bisect_right(starts, start) - 1, 0)
        while i < len(sentences) and sentences[i].start < end:
            s = sentences[i]
            overlap = min(end, s.end) - max(start, s.start)
            if overlap > 0:
                if overlap >= threshold * min(s.end - s.start, end - start):
                    positive[i].add(label)
                    hit = True
                else:
                    ambiguous[i].add(label)
            i += 1
        covered.append(hit)
    for pos, amb in zip(positive, ambiguous, strict=True):
        amb -= pos
    return positive, ambiguous, covered


def prepare_contract() -> pd.DataFrame:
    cfg = load_module_config("contract")
    root = cuad_root()
    if not (root / "CUADv1.json").exists():
        raise FileNotFoundError(f"{root} not found; run `clauseguard data download contract` first")
    label_of = {spec.source: spec.id for spec in cfg.labels}
    order = {label: i for i, label in enumerate(cfg.label_ids)}
    threshold = float(cfg.data["overlap_threshold"])
    test_titles = _titles(root / "test.json")
    train_titles = _titles(root / "train_separate_questions.json")

    stats: Counter = Counter()
    rows, span_rows = [], []
    for doc in json.loads((root / "CUADv1.json").read_text(encoding="utf-8"))["data"]:
        title = doc["title"]
        context = doc["paragraphs"][0]["context"]
        tm = clean_text(context)
        text = tm.text
        spans = []
        for qa in doc["paragraphs"][0]["qas"]:
            category = qa["id"].split("__")[-1]
            if category not in label_of:
                continue
            for answer in qa["answers"]:
                start, raw_text = answer["answer_start"], answer["text"]
                if context[start : start + len(raw_text)] != raw_text:
                    stats["offset_mismatch"] += 1
                    continue
                s, e = tm.from_source_span(start, start + len(raw_text))
                if e > s:
                    spans.append((label_of[category], s, e))
        spans = sorted(set(spans), key=lambda x: (x[1], x[2], x[0]))
        sentences = [
            s for s in split_sentences(text) if is_classifiable(s.text, cfg.min_unit_words)
        ]
        positive, ambiguous, covered = label_sentences(sentences, spans, threshold)

        for (label, s, e), hit in zip(spans, covered, strict=True):
            stats[f"spans:{label}"] += 1
            stats[f"covered:{label}"] += hit
            span_rows.append({"doc_id": title, "label": label, "start": s, "end": e})
        for i, (sentence, pos, amb) in enumerate(zip(sentences, positive, ambiguous, strict=True)):
            rows.append(
                {
                    "doc_id": title,
                    "unit_id": f"{title}:{i}",
                    "unit_type": "sentence",
                    "text": sentence.text,
                    "char_start": sentence.start,
                    "char_end": sentence.end,
                    "labels": sorted(pos, key=order.__getitem__),
                    "exclude_from_training": bool(amb),
                    "meta": json.dumps({"ambiguous": sorted(amb)} if amb else {}),
                }
            )
        stats["contracts"] += 1
        stats["sentences"] += len(sentences)
        stats["ambiguous_sentences"] += sum(bool(a) for a in ambiguous)

    df = pd.DataFrame(rows)
    unknown = set(df["doc_id"]) - test_titles - train_titles
    if unknown:
        raise ValueError(f"{len(unknown)} contracts are in neither the train nor the test file")
    label_sets: dict[str, set[str]] = defaultdict(set)
    for title, labels in zip(df["doc_id"], df["labels"], strict=True):
        if title in train_titles:
            label_sets[title].update(labels)
    for title in train_titles:
        label_sets.setdefault(title, set())
    val_ratio = float(cfg.data["validation_ratio"])
    split_of = load_or_create_split(
        "contract",
        dict(label_sets),
        {"train": 1 - val_ratio, "validation": val_ratio},
        cfg.data["split_seed"],
    )
    split_of.update(dict.fromkeys(test_titles, "test"))
    df["split"] = df["doc_id"].map(split_of)
    df["char_start"] = df["char_start"].astype("Int64")
    df["char_end"] = df["char_end"].astype("Int64")
    df = df[COLUMNS]
    validate_units(df, cfg.label_ids)

    out = processed_dir("contract")
    out.mkdir(parents=True, exist_ok=True)
    spans_df = pd.DataFrame(span_rows)
    spans_df["split"] = spans_df["doc_id"].map(split_of)
    spans_df.to_parquet(out / "spans.parquet", index=False)
    (out / "conversion_stats.json").write_text(json.dumps(dict(stats), indent=2, sort_keys=True))
    return df


def card_extras(df: pd.DataFrame) -> tuple[list[str], dict]:
    cfg = load_module_config("contract")
    stats = json.loads((processed_dir("contract") / "conversion_stats.json").read_text())
    y = label_matrix(df["labels"], cfg.label_ids)
    md = ["## Conversion (CUAD spans → sentences)", ""]
    md.append(
        f"{stats['contracts']} contracts, {stats['sentences']:,} sentences. A sentence is labelled "
        f"when it overlaps a span by at least {cfg.data['overlap_threshold']} × the shorter of "
        f"the two. {stats.get('ambiguous_sentences', 0):,} sentences overlap a span only "
        "slightly; they are excluded from training and kept for evaluation. "
        f"Offset mismatches in the source: {stats.get('offset_mismatch', 0)}."
    )
    md += ["", "Span coverage: share of gold spans with at least one positive sentence.", ""]
    rows, total, hits = [], 0, 0
    for j, label in enumerate(cfg.label_ids):
        n, c = stats.get(f"spans:{label}", 0), stats.get(f"covered:{label}", 0)
        total, hits = total + n, hits + c
        contracts = df.loc[y[:, j] == 1, "doc_id"].nunique()
        rows.append([f"`{label}`", n, f"{c / n:.1%}" if n else "–", int(y[:, j].sum()), contracts])
    md += md_table(["label", "gold spans", "covered", "positive sentences", "contracts"], rows)
    md.append(f"\n**Overall span coverage: {hits / max(total, 1):.2%}** ({hits:,} of {total:,}).")

    rng_df = df[y.any(axis=1)].sample(70, random_state=0)
    review = pd.concat([rng_df, df[~y.any(axis=1)].sample(30, random_state=0)])
    path = data_cards_dir() / "contract_review_sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    review.assign(labels=review["labels"].map("; ".join), correct="", notes="")[
        ["unit_id", "text", "labels", "correct", "notes"]
    ].to_csv(path, index=False)
    md += [
        "",
        "## Manual check",
        "",
        f"100 random sentences (70 labelled, 30 unlabelled) are in `{path.name}`. Fill in the "
        "`correct` column (y/n) and record the error rate here.",
        "",
    ]
    return md, {"conversion": stats, "span_coverage": hits / max(total, 1)}
