"""Document-level splits that keep rare labels spread across splits.

Iterative stratification (Sechidis et al., 2011) over documents: repeatedly take the label with
the fewest unassigned documents and give each of those documents to the split that most needs
that label. Assignments are saved to ``data/splits/<name>.json`` and reused, so the split never
changes silently.
"""

from __future__ import annotations

import json
import random
from collections import Counter

from clauseguard.paths import splits_dir


def stratified_group_split(
    label_sets: dict[str, set[str]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign every document (key of ``label_sets``) to one of the splits in ``ratios``."""
    rng = random.Random(seed)
    docs = sorted(label_sets)
    rng.shuffle(docs)
    total = sum(ratios.values())
    want_size = {s: r / total * len(docs) for s, r in ratios.items()}
    label_counts = Counter(label for d in docs for label in label_sets[d])
    want_label = {
        s: {lab: c * r / total for lab, c in label_counts.items()} for s, r in ratios.items()
    }
    order = {s: i for i, s in enumerate(ratios)}
    assignment: dict[str, str] = {}
    remaining = list(docs)
    while remaining:
        counts = Counter(label for d in remaining for label in label_sets[d])
        label = min(counts, key=lambda lab: (counts[lab], lab)) if counts else None
        batch = [d for d in remaining if label is None or label in label_sets[d]]
        for doc in batch:
            # the split that most needs this label, then the most room, then random
            split = max(
                ratios,
                key=lambda s, lab=label: (
                    want_label[s][lab] if lab is not None else 0.0,
                    want_size[s],
                    rng.random(),
                    -order[s],
                ),
            )
            assignment[doc] = split
            want_size[split] -= 1
            for lab in label_sets[doc]:
                want_label[split][lab] -= 1
        remaining = [d for d in remaining if d not in assignment]
    return assignment


def load_or_create_split(
    name: str, label_sets: dict[str, set[str]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Reuse ``data/splits/<name>.json`` if it covers exactly these documents."""
    path = splits_dir() / f"{name}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if set(saved["assignment"]) == set(label_sets):
            return saved["assignment"]
        raise ValueError(
            f"{path} does not match the current documents; delete it to create a new split"
        )
    assignment = stratified_group_split(label_sets, ratios, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"seed": seed, "ratios": ratios, "assignment": dict(sorted(assignment.items()))},
            indent=1,
        )
    )
    return assignment
