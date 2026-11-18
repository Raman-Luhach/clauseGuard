"""Document-type router: is this a terms of service, a privacy policy or a contract?

A TF-IDF + logistic regression classifier over windows of a document's text. The three document
families use different vocabulary, so the task is easy; what matters is knowing when the router
is unsure. It suggests a type and the user can always change it: the UI pre-selects a confident
guess, and leaves the choice to the user otherwise.

Training documents are the full texts behind the three datasets: the 50 CLAUDETTE terms of
service, the 115 OPP-115 privacy policies and the 510 CUAD contracts. Each document keeps the
split of its module (UNFAIR-ToS, OPP-115 and CUAD splits), so no test document is seen in
training. A fourth class, ``other``, is unrelated text (news articles from AG News, Wikipedia
articles from WikiText-2): without it, a news article still gets a confident legal type.
Training samples random windows of 150-600 words, so the router copes with excerpts and with
web pages that start with menus; prediction averages windows spread over the text.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from clauseguard.data.card import md_table
from clauseguard.errors import ModelNotAvailableError
from clauseguard.ingestion.clean import clean_text
from clauseguard.paths import _CODE_ROOT, configs_dir, raw_dir, reports_dir, root, units_path

OTHER = "other"  # not a document type: the text looks unrelated to all three
OTHER_DOC_WORDS = 600  # news articles are short: several are joined into one document
WINDOW_WORDS = 400  # prediction window
MAX_WINDOWS = 5  # windows averaged per document at prediction time
TRAIN_WINDOW_RANGE = (150, 600)
WINDOWS_PER_CLASS = 1200  # training windows per document type (balanced classes)
MIN_CONFIDENCE = 0.60  # below this, the router does not pre-select a type
C_GRID = (1.0, 4.0, 16.0, 64.0, 256.0)
SEED = 13


def router_dir() -> Path:
    return root() / "artifacts" / "router"


# ---------------------------------------------------------------- training documents


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def _module_splits(module: str) -> dict[str, str]:
    units = pd.read_parquet(units_path(module), columns=["doc_id", "split"])
    return units.drop_duplicates("doc_id").set_index("doc_id")["split"].to_dict()


def _tos_documents() -> list[dict]:
    folder = raw_dir("claudette") / "ToS" / "OriginalTaggedDocuments"
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} not found; run `clauseguard data download claudette` (the router "
            "needs the full terms of service)"
        )
    splits = _module_splits("tos")
    return [
        {"doc_id": p.stem, "type": "tos", "split": splits.get(p.stem, "train"), "text": text}
        for p in sorted(folder.glob("*.xml"))
        if (text := _strip_tags(p.read_text(encoding="utf-8", errors="replace")))
    ]


def _privacy_documents() -> list[dict]:
    from clauseguard.data.privacy import opp_root

    splits = _module_splits("privacy")
    docs = []
    for p in sorted((opp_root() / "sanitized_policies").glob("*.html")):
        text = _strip_tags(p.read_text(encoding="utf-8", errors="replace").replace("|||", "\n\n"))
        docs.append({"doc_id": p.stem, "type": "privacy", "split": splits[p.stem], "text": text})
    return docs


def _contract_documents() -> list[dict]:
    from clauseguard.data.contract import cuad_root

    splits = _module_splits("contract")
    data = json.loads((cuad_root() / "CUADv1.json").read_text(encoding="utf-8"))["data"]
    return [
        {
            "doc_id": d["title"],
            "type": "contract",
            "split": splits[d["title"]],
            "text": d["paragraphs"][0]["context"],
        }
        for d in data
    ]


def _split_of(index: int) -> str:
    """A fixed 70/15/15 split for the "other" documents, by position."""
    return (("train",) * 14 + ("validation",) * 3 + ("test",) * 3)[index % 20]


def _other_documents() -> list[dict]:
    """News articles joined into ~600-word documents, and Wikipedia articles."""
    news_path, wiki_path = raw_dir("ag_news") / "test.parquet", raw_dir("wikitext") / "test.parquet"
    if not (news_path.exists() and wiki_path.exists()):
        raise FileNotFoundError(
            "the router's unrelated-text data is missing; run `clauseguard data download router`"
        )
    docs, current, n_words = [], [], 0
    for text in pd.read_parquet(news_path)["text"]:
        current.append(text)
        n_words += len(text.split())
        if n_words >= OTHER_DOC_WORDS:
            docs.append(("news", "\n\n".join(current)))
            current, n_words = [], 0
    article: list[str] = []
    for line in [*pd.read_parquet(wiki_path)["text"], " = END = \n"]:
        if re.fullmatch(r" = [^=].* = \n", line):  # a new top-level article
            if len(" ".join(article).split()) >= 150:
                docs.append(("wiki", " ".join(article)))
            article = []
        elif line.strip() and not line.startswith(" = "):
            # undo WikiText's tokenisation ("film , television" -> "film, television")
            article.append(
                re.sub(r" @(.)@ ", r"\1", line).replace(" , ", ", ").replace(" . ", ". ")
            )
    return [
        {"doc_id": f"{kind}-{i}", "type": OTHER, "split": _split_of(i), "text": text}
        for i, (kind, text) in enumerate(docs)
    ]


def router_documents() -> pd.DataFrame:
    """One row per training document: doc_id, type, split, cleaned text."""
    rows = _tos_documents() + _privacy_documents() + _contract_documents() + _other_documents()
    for row in rows:
        row["text"] = clean_text(row["text"]).text
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- windows and model


def windows(text: str, size: int = WINDOW_WORDS, n: int = MAX_WINDOWS) -> list[str]:
    """Up to ``n`` windows of ``size`` words, evenly spread; the first starts at the top."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    starts = np.linspace(0, len(words) - size, num=min(n, len(words) // size + 1)).astype(int)
    return [" ".join(words[s : s + size]) for s in dict.fromkeys(starts.tolist())]


def sample_windows(docs: pd.DataFrame, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Random windows for training, the same number per document type."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for doc_type, group in docs.groupby("type"):
        per_doc = max(2, int(np.ceil(WINDOWS_PER_CLASS / len(group))))
        for text in group["text"]:
            words = text.split()
            for _ in range(per_doc):
                size = int(rng.integers(*TRAIN_WINDOW_RANGE))
                start = int(rng.integers(0, max(1, len(words) - size)))
                texts.append(" ".join(words[start : start + size]))
                labels.append(doc_type)
    return texts, labels


def _pipeline(c: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only: no numbers, no markup
        ),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
    )


@dataclass
class Detection:
    document_type: str | None  # the suggestion, or None when the router is unsure
    confidence: float  # probability of the most likely class (may be "other")
    probabilities: dict[str, float]  # per class, including "other"
    confident: bool  # a document type is suggested
    unrelated: bool = False  # most likely "other": not a ToS, privacy policy or contract
    model: str = "router"

    @property
    def best(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def as_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "confident": self.confident,
            "unrelated": self.unrelated,
            "model": self.model,
        }


class Router:
    def __init__(self, pipeline, classes: list[str], min_confidence: float, meta: dict):
        self.pipeline = pipeline
        self.classes = classes
        self.min_confidence = min_confidence
        self.meta = meta

    def probabilities(self, text: str) -> np.ndarray:
        probs = self.pipeline.predict_proba(windows(text))
        return probs.mean(axis=0)

    def detect(self, text: str) -> Detection:
        probs = self.probabilities(text)
        best = self.classes[int(np.argmax(probs))]
        confidence = float(probs.max())
        confident = best != OTHER and confidence >= self.min_confidence
        return Detection(
            document_type=best if confident else None,
            confidence=confidence,
            probabilities={c: float(p) for c, p in zip(self.classes, probs, strict=True)},
            confident=confident,
            unrelated=best == OTHER,
            model=self.meta.get("version", "router"),
        )

    def save(self, path: Path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "model.joblib")
        meta = {**self.meta, "classes": self.classes, "min_confidence": self.min_confidence}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, path: Path | None = None) -> Router:
        import joblib

        path = path or router_dir()
        if not (path / "model.joblib").exists():
            raise ModelNotAvailableError(
                "no document-type router; train it with `clauseguard router train`"
            )
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            joblib.load(path / "model.joblib"), meta["classes"], meta["min_confidence"], meta
        )


@lru_cache(maxsize=1)
def get_router() -> Router:
    return Router.load()


def detect_document_type(text: str) -> Detection:
    return get_router().detect(text)


# ---------------------------------------------------------------- training and evaluation


def _evaluate_docs(router: Router, docs: pd.DataFrame) -> dict:
    detections = [router.detect(t) for t in docs["text"]]
    gold = docs["type"].tolist()
    best = [d.best for d in detections]
    # "confident" = the router commits to an answer: a legal type, or "unrelated"
    confident = np.array([d.confident or d.unrelated for d in detections])
    correct = np.array([b == g for b, g in zip(best, gold, strict=True)])
    confusion = pd.crosstab(pd.Series(gold, name="gold"), pd.Series(best, name="predicted"))
    return {
        "documents": len(docs),
        "accuracy": float(correct.mean()),
        "confident_share": float(confident.mean()),
        "accuracy_when_confident": float(correct[confident].mean()) if confident.any() else 0.0,
        "wrong_and_confident": int((~correct & confident).sum()),
        "confusion": {g: row.to_dict() for g, row in confusion.iterrows()},
        "mistakes": [
            {"doc_id": d, "gold": g, "predicted": b, "confidence": round(det.confidence, 3)}
            for d, g, b, det in zip(docs["doc_id"], gold, best, detections, strict=True)
            if b != g
        ],
    }


def _excerpt_accuracy(router: Router, docs: pd.DataFrame, seed: int = SEED) -> dict:
    """Short excerpts (100-200 words) from random places: harder than whole documents."""
    rng = np.random.default_rng(seed)
    hits, confident = [], []
    for text, gold in zip(docs["text"], docs["type"], strict=True):
        words = text.split()
        for _ in range(3):
            size = int(rng.integers(100, 200))
            start = int(rng.integers(0, max(1, len(words) - size)))
            d = router.detect(" ".join(words[start : start + size]))
            hits.append(d.best == gold)
            confident.append(d.confident or d.unrelated)
    hits_a, conf_a = np.array(hits), np.array(confident)
    return {
        "excerpts": len(hits),
        "accuracy": float(hits_a.mean()),
        "confident_share": float(conf_a.mean()),
        "accuracy_when_confident": float(hits_a[conf_a].mean()) if conf_a.any() else 0.0,
    }


def _out_of_scope(router: Router) -> dict:
    """Unrelated text (hand-written news paragraphs) should not get a confident legal type."""
    paragraphs = [
        p.strip()
        for p in (configs_dir() / "robustness" / "out_of_scope.txt").read_text().split("\n\n")
        if p.strip()
    ]
    texts = paragraphs + [" ".join(paragraphs)]
    detections = [router.detect(t) for t in texts]
    return {
        "texts": len(texts),
        "confident": int(sum(d.confident for d in detections)),
        "called_unrelated": int(sum(d.unrelated for d in detections)),
        "max_confidence": float(max(d.confidence for d in detections)),
    }


def _examples(router: Router) -> dict:
    """The sample documents in examples/ (real documents, not in any training set)."""
    from clauseguard.evaluation.robustness import EXAMPLES

    out = {}
    for doc_type, name in EXAMPLES.items():
        d = router.detect(clean_text((_CODE_ROOT / "examples" / name).read_text()).text)
        out[name] = {"expected": doc_type, **d.as_dict()}
    return out


def train_router(log: Callable[[str], None] = print) -> dict:
    """Train on the train split, choose C on validation, report test; save to artifacts."""
    from datetime import UTC, datetime

    docs = router_documents()
    counts = docs.groupby(["type", "split"]).size().unstack(fill_value=0)
    log(f"router documents:\n{counts}")
    train, val, test = (docs[docs["split"] == s] for s in ("train", "validation", "test"))
    texts, labels = sample_windows(train)
    log(f"{len(texts)} training windows")
    best = None
    for c in C_GRID:
        pipeline = _pipeline(c).fit(texts, labels)
        router = Router(pipeline, list(pipeline.classes_), MIN_CONFIDENCE, {})
        val_metrics = _evaluate_docs(router, val)
        excerpt = _excerpt_accuracy(router, val)
        # Whole documents first; then the share of excerpts answered confidently *and*
        # correctly (a router that is right but rarely sure is not much help).
        score = (
            round(val_metrics["accuracy"], 3),
            round(excerpt["accuracy_when_confident"] * excerpt["confident_share"], 3),
        )
        log(
            f"C={c}: validation documents {score[0]:.3f}; excerpts confident and correct "
            f"{score[1]:.3f} (accuracy {excerpt['accuracy']:.3f})"
        )
        if best is None or score > best[0]:
            best = (score, c, router)
    assert best is not None
    _, c, router = best
    meta = {
        "version": f"router-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "C": c,
        "training_windows": len(texts),
        "documents": {str(k): v for k, v in counts.to_dict(orient="index").items()},
        "validation": _evaluate_docs(router, val),
        "validation_excerpts": _excerpt_accuracy(router, val),
        "test": _evaluate_docs(router, test),
        "test_excerpts": _excerpt_accuracy(router, test),
        "out_of_scope": _out_of_scope(router),
        "examples": _examples(router),
    }
    router.meta = meta
    router.save(router_dir())
    get_router.cache_clear()
    write_router_report(meta)
    log(
        f"router: C={c}; test documents {meta['test']['accuracy']:.3f}, test excerpts "
        f"{meta['test_excerpts']['accuracy']:.3f}; out-of-scope texts called confidently: "
        f"{meta['out_of_scope']['confident']}/{meta['out_of_scope']['texts']}"
    )
    return meta


def write_router_report(meta: dict) -> Path:
    md = [
        "# Document-type router",
        "",
        f"TF-IDF + logistic regression (C={meta['C']}), {meta['training_windows']} training "
        f"windows. A type is pre-selected only when its probability is at least "
        f"{MIN_CONFIDENCE:.2f}; otherwise the user chooses.",
        "",
        "## Documents per split",
        "",
    ]
    md += md_table(
        ["type", "train", "validation", "test"],
        [
            [t, v.get("train", 0), v.get("validation", 0), v.get("test", 0)]
            for t, v in meta["documents"].items()
        ],
    )
    md += ["", "## Results", ""]
    rows = []
    for name in ("validation", "validation_excerpts", "test", "test_excerpts"):
        m = meta[name]
        rows.append(
            [
                name.replace("_", " "),
                m.get("documents", m.get("excerpts")),
                f"{m['accuracy']:.3f}",
                f"{m['confident_share']:.3f}",
                f"{m['accuracy_when_confident']:.3f}",
            ]
        )
    md += md_table(["set", "n", "accuracy", "confident share", "accuracy when confident"], rows)
    oos = meta["out_of_scope"]
    md += [
        "",
        "Excerpts are 100-200-word windows from random places in each document.",
        "",
        f"Hand-written unrelated text (news paragraphs, {oos['texts']} texts): "
        f"{oos['confident']} got a legal document type; {oos['called_unrelated']} were "
        "recognised as unrelated.",
        "",
        "## Example documents (not in any training set)",
        "",
    ]
    md += md_table(
        ["file", "expected", "suggested", "confidence"],
        [
            [name, e["expected"], e["document_type"] or "(unsure)", f"{e['confidence']:.2f}"]
            for name, e in meta["examples"].items()
        ],
    )
    mistakes = meta["test"]["mistakes"]
    if mistakes:
        md += ["", "## Test mistakes", ""]
        md += md_table(
            ["document", "gold", "predicted", "confidence"],
            [[m["doc_id"], m["gold"], m["predicted"], m["confidence"]] for m in mistakes],
        )
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "router.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
