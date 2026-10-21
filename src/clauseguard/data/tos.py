"""UNFAIR-ToS → ToS unit table.

UNFAIR-ToS is already sentence-level with official per-document splits, so no label conversion
is needed. LexGLUE dropped the document names; when the original CLAUDETTE corpus is present we
put them back by aligning the two versions sentence by sentence (they contain the same 9,414
sentences in the same order), and we cross-check every label against CLAUDETTE's own files.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from clauseguard.config import load_module_config
from clauseguard.data.card import md_table
from clauseguard.data.units import SPLITS, UNKNOWN_DOC, validate_units
from clauseguard.ingestion.normalize import unfair_tos_style
from clauseguard.paths import raw_dir

EXPECTED_ROWS = {"train": 5532, "validation": 2275, "test": 1607}
_PTB = {"-lrb-": "(", "-rrb-": ")", "-lsb-": "[", "-rsb-": "]", "-lcb-": "{", "-rcb-": "}"}
# CLAUDETTE label folder for each LexGLUE label index.
CLAUDETTE_FOLDERS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "A"]
MIN_ALIGNMENT = 0.95


def _align_key(text: str) -> str:
    return " ".join(_PTB.get(w, w) for w in text.split())


def _claudette_root() -> Path | None:
    root = raw_dir("claudette") / "ToS"
    return root if (root / "Sentences").is_dir() else None


def align_with_claudette(texts: list[str], root: Path) -> tuple[list[str | None], list[int | None]]:
    """Return (document name, line number) for each LexGLUE sentence, or None when unmatched."""
    keys = [_align_key(t) for t in texts]
    starts: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        starts[key].append(i)

    doc_of: list[str | None] = [None] * len(texts)
    line_of: list[int | None] = [None] * len(texts)
    for path in sorted((root / "Sentences").glob("*.txt")):
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        sentences = [_align_key(line) for line in lines if line.strip()]
        if not sentences:
            continue
        best_pos, best_hits = None, 0
        for pos in starts.get(sentences[0], []):
            window = keys[pos : pos + len(sentences)]
            hits = sum(a == b for a, b in zip(window, sentences, strict=False))
            if hits > best_hits:
                best_pos, best_hits = pos, hits
        if best_pos is None or best_hits / len(sentences) < MIN_ALIGNMENT:
            continue
        for offset in range(min(len(sentences), len(texts) - best_pos)):
            doc_of[best_pos + offset] = path.stem
            line_of[best_pos + offset] = offset
    return doc_of, line_of


def claudette_label_agreement(
    df: pd.DataFrame, root: Path, label_ids: list[str]
) -> dict[str, float]:
    """Share of aligned sentences where LexGLUE and CLAUDETTE agree, per label."""
    agree: Counter[str] = Counter()
    total: Counter[str] = Counter()
    cache: dict[tuple[str, str], list[int]] = {}
    for doc, line, labels in zip(
        df["doc_id"], df["meta"].map(json.loads), df["labels"], strict=True
    ):
        if doc == UNKNOWN_DOC:
            continue
        for label_id, folder in zip(label_ids, CLAUDETTE_FOLDERS, strict=True):
            key = (folder, doc)
            if key not in cache:
                path = root / f"Labels_{folder}" / f"{doc}.txt"
                cache[key] = [int(v) for v in path.read_text().split()] if path.exists() else []
            values = cache[key]
            pos = line["claudette_line"]
            if pos is None or pos >= len(values):
                continue
            total[label_id] += 1
            agree[label_id] += (values[pos] == 1) == (label_id in labels)
    return {label: agree[label] / total[label] for label in label_ids if total[label]}


def claudette_fairness_levels(root: Path) -> dict[str, dict[str, int]]:
    """Counts of expert fairness levels per category in CLAUDETTE's tagged documents
    (1 = clearly fair, 2 = potentially unfair, 3 = clearly unfair)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in (root / "OriginalTaggedDocuments").glob("*.xml"):
        for tag, level in re.findall(
            r"<(ltd|ter|ch|cr|use|law|j|a)(\d)>", path.read_text(errors="replace")
        ):
            counts[tag.upper()][level] += 1
    return {folder: dict(sorted(counts[folder].items())) for folder in CLAUDETTE_FOLDERS}


def card_extras(df: pd.DataFrame) -> tuple[list[str], dict]:
    cfg = load_module_config("tos")
    root = _claudette_root()
    md = ["## Document names (CLAUDETTE alignment)", ""]
    aligned = int((df["doc_id"] != UNKNOWN_DOC).sum())
    md.append(
        f"{aligned} of {len(df)} sentences were matched to one of the 50 original CLAUDETTE "
        f"documents ({df.loc[df['doc_id'] != UNKNOWN_DOC, 'doc_id'].nunique()} documents found)."
    )
    stats: dict = {"claudette_aligned": aligned}
    if root is None:
        md.append(
            "CLAUDETTE is not downloaded, so document names and the checks below are missing."
        )
        return md, stats

    agreement = claudette_label_agreement(df, root, cfg.label_ids)
    md += [
        "",
        "Label cross-check: share of aligned sentences where LexGLUE and CLAUDETTE agree.",
        "",
    ]
    md += md_table(["label", "agreement"], [[f"`{k}`", f"{v:.2%}"] for k, v in agreement.items()])
    stats["claudette_label_agreement"] = agreement

    levels = claudette_fairness_levels(root)
    md += [
        "",
        "## Expert fairness ratings (CLAUDETTE tagged documents)",
        "",
        "Level 1 = clearly fair, 2 = potentially unfair, 3 = clearly unfair. Counts are clause "
        "tags, not sentences. Useful evidence for the default severities in taxonomy.yaml.",
        "",
    ]
    rows = [
        [f"`{label}`", *(levels[folder].get(level, 0) for level in "123")]
        for label, folder in zip(cfg.label_ids, CLAUDETTE_FOLDERS, strict=True)
    ]
    md += md_table(["label", "level 1", "level 2", "level 3"], rows)
    stats["claudette_fairness_levels"] = levels
    return md, stats


def prepare_tos() -> pd.DataFrame:
    cfg = load_module_config("tos")
    src = raw_dir("unfair_tos")
    frames = []
    for split in SPLITS:
        path = src / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `clauseguard data download tos` first")
        part = pd.read_parquet(path)
        if len(part) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, found {len(part)}")
        part = part.assign(split=split, source_index=range(len(part)))
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    n_labels = len(cfg.labels)
    bad = [labels for labels in df["labels"] for i in labels if not 0 <= i < n_labels]
    if bad:
        raise ValueError(f"label index out of range in {len(bad)} rows")

    root = _claudette_root()
    if root is not None:
        doc_of, line_of = align_with_claudette(df["text"].tolist(), root)
    else:
        doc_of, line_of = [None] * len(df), [None] * len(df)

    label_ids = cfg.label_ids
    units = pd.DataFrame(
        {
            "doc_id": [d or UNKNOWN_DOC for d in doc_of],
            "unit_id": [
                f"{d}:{ln}" if d is not None else f"tos-{s}-{i}"
                for d, ln, s, i in zip(
                    doc_of, line_of, df["split"], df["source_index"], strict=True
                )
            ],
            "unit_type": "sentence",
            "text": df["text"].map(unfair_tos_style),
            "char_start": pd.array([None] * len(df), dtype="Int64"),
            "char_end": pd.array([None] * len(df), dtype="Int64"),
            "labels": [[label_ids[i] for i in sorted(set(labels))] for labels in df["labels"]],
            "split": df["split"],
            "exclude_from_training": False,
            "meta": [
                json.dumps({"source_index": int(i), "claudette_line": ln})
                for i, ln in zip(df["source_index"], line_of, strict=True)
            ],
        }
    )
    validate_units(units, label_ids)
    return units
