"""Zero-shot LLM classification vs. the fine-tuned classifier, on the same sample of units.

Answers the report's question "when is a fine-tuned model enough?" with numbers: per-label
F1, run-to-run consistency (the LLM is asked twice), tokens used and latency. Uses the same
Claude settings as the explanation layer (CLAUSEGUARD_LLM_MODEL, CLAUSEGUARD_LLM_EFFORT, ...);
every call costs money, so the sample is small and the command must be run deliberately.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np

from clauseguard.config import load_module_config, load_taxonomy
from clauseguard.data.card import md_table
from clauseguard.data.units import label_matrix, load_units
from clauseguard.evaluation.metrics import compute_metrics
from clauseguard.explain.llm import SERVER_FALLBACK_MODELS, LLMSettings
from clauseguard.models.base import load_model, serving_model_dir
from clauseguard.models.runs import read_json, resolve_run, write_json
from clauseguard.models.train import risky_mask, subsample
from clauseguard.paths import reports_dir

BATCH = 20

SYSTEM = """You label units of text from legal documents. For each unit, return every category \
from the list that the unit expresses; return an empty list when none applies. Use only the \
category ids given, and label each unit independently."""


def _schema(label_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "labels": {"type": "array", "items": {"type": "string", "enum": label_ids}},
                    },
                    "required": ["id", "labels"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _classify(client, settings: LLMSettings, texts: list[str], label_ids: list[str]) -> tuple:
    taxonomy = load_taxonomy()
    categories = [
        {"id": label, "definition": taxonomy.labels[label].definition} for label in label_ids
    ]
    params = {
        "model": settings.model,
        "max_tokens": 16000,
        "system": SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "categories": categories,
                        "units": [{"id": i, "text": t} for i, t in enumerate(texts)],
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": _schema(label_ids)},
        },
    }
    if settings.model in SERVER_FALLBACK_MODELS:
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"
    response = client.beta.messages.create(**params)
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(block.text for block in response.content if block.type == "text")
    labels = {int(item["id"]): item["labels"] for item in json.loads(text)["items"]}
    return labels, response.usage.input_tokens, response.usage.output_tokens


def llm_compare(
    module: str,
    *,
    n: int = 200,
    split: str = "validation",
    final: bool = False,
    repeats: int = 2,
    seed: int = 42,
    settings: LLMSettings | None = None,
    client=None,
    log: Callable[[str], None] = print,
) -> dict:
    if split == "test" and not final:
        raise ValueError("the test set is for final results only; pass --final")
    settings = settings or LLMSettings.from_env()
    if client is None:
        import anthropic

        client = anthropic.Anthropic(timeout=max(settings.timeout, 120.0), max_retries=2)
    cfg = load_module_config(module)
    label_ids = cfg.label_ids
    units = load_units(module)
    sample = subsample(units[units["split"] == split], label_ids, n, seed).reset_index(drop=True)
    texts = sample["text"].tolist()
    y = label_matrix(sample["labels"], label_ids)

    runs, tokens_in, tokens_out, seconds = [], 0, 0, 0.0
    for r in range(repeats):
        pred = np.zeros_like(y)
        started = time.perf_counter()
        for start in range(0, len(texts), BATCH):
            labels, t_in, t_out = _classify(
                client, settings, texts[start : start + BATCH], label_ids
            )
            tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out
            for local, chosen in labels.items():
                for label in chosen:
                    if 0 <= local < BATCH and start + local < len(texts) and label in label_ids:
                        pred[start + local, label_ids.index(label)] = 1
        seconds += time.perf_counter() - started
        runs.append(pred)
        log(f"LLM run {r + 1}/{repeats} done")
    llm_metrics = [
        compute_metrics(
            y, p.astype(float), [0.5] * len(label_ids), label_ids, risky=risky_mask(cfg)
        )
        for p in runs
    ]
    consistency = (
        float(np.mean([(runs[0] == p).all(axis=1).mean() for p in runs[1:]]))
        if len(runs) > 1
        else float("nan")
    )

    run_path = resolve_run(module, None, cfg.selection_metric)
    model = load_model(serving_model_dir(run_path))
    by_label = read_json(run_path / "thresholds.json")
    started = time.perf_counter()
    p = model.predict_proba(texts)
    classifier_seconds = time.perf_counter() - started
    classifier = compute_metrics(
        y, p, [by_label[lab] for lab in label_ids], label_ids, risky=risky_mask(cfg)
    )

    report = {
        "module": module,
        "split": split,
        "n_units": len(texts),
        "llm_model": settings.model,
        "llm_effort": settings.effort,
        "classifier_run": run_path.name,
        "llm": {
            "macro_f1": [m["macro_f1"] for m in llm_metrics],
            "micro_f1": [m["micro"]["f1"] for m in llm_metrics],
            "any_risk_recall": [m["any_risk"]["recall"] for m in llm_metrics],
            "consistency_identical_label_sets": consistency,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "ms_per_unit": round(1000 * seconds / (len(texts) * repeats), 1),
            "per_label_f1": {lab: llm_metrics[0]["per_label"][lab]["f1"] for lab in label_ids},
        },
        "classifier": {
            "macro_f1": classifier["macro_f1"],
            "micro_f1": classifier["micro"]["f1"],
            "any_risk_recall": classifier["any_risk"]["recall"],
            "ms_per_unit": round(1000 * classifier_seconds / len(texts), 2),
            "per_label_f1": {lab: classifier["per_label"][lab]["f1"] for lab in label_ids},
        },
    }
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{module}_llm_compare.json", report)
    llm, clf = report["llm"], report["classifier"]
    md = [
        f"# Zero-shot LLM vs. fine-tuned classifier: {module}",
        "",
        f"{len(texts)} {split} units (half of them labelled). LLM: `{settings.model}` (effort "
        f"{settings.effort}), asked {repeats} times. Classifier: `{run_path.name}`.",
        "",
    ]
    md += md_table(
        ["", "LLM (zero-shot)", "classifier"],
        [
            ["macro F1", " / ".join(f"{v:.3f}" for v in llm["macro_f1"]), f"{clf['macro_f1']:.3f}"],
            ["micro F1", " / ".join(f"{v:.3f}" for v in llm["micro_f1"]), f"{clf['micro_f1']:.3f}"],
            [
                "any-risk recall",
                " / ".join(f"{v:.3f}" for v in llm["any_risk_recall"]),
                f"{clf['any_risk_recall']:.3f}",
            ],
            ["identical answers across runs", f"{consistency:.1%}", "100% (deterministic)"],
            ["time per unit", f"{llm['ms_per_unit']} ms", f"{clf['ms_per_unit']} ms"],
            ["tokens (all runs)", f"{tokens_in:,} in / {tokens_out:,} out", "–"],
        ],
    )
    (out / f"{module}_llm_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
