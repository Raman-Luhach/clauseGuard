"""Fine-tuned transformer for multi-label classification.

A small custom training loop (rather than the Hugging Face Trainer) keeps the important choices
explicit: class-weighted BCE for rare labels, linear warm-up and decay, bf16/fp16 on CUDA, and
early stopping on validation macro PR-AUC, which does not depend on a 0.5 threshold.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from clauseguard.config import ModelSpec, PosWeightSpec
from clauseguard.evaluation.metrics import macro_pr_auc
from clauseguard.models.base import MultiLabelModel


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pos_weights(y: np.ndarray, spec: PosWeightSpec) -> torch.Tensor:
    pos = y.sum(axis=0).astype(float)
    neg = len(y) - pos
    ratio = np.divide(neg, pos, out=np.ones_like(pos), where=pos > 0)
    weights = {"none": np.ones_like(ratio), "sqrt": np.sqrt(ratio), "linear": ratio}[spec.mode]
    return torch.tensor(np.clip(weights, 1.0, spec.max), dtype=torch.float32)


class TransformerModel(MultiLabelModel):
    kind = "transformer"

    def __init__(self, labels: list[str], tokenizer, model, max_length: int, device: torch.device):
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.max_length = max_length
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str, labels: list[str], max_length: int, device: torch.device
    ) -> TransformerModel:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Set the label map on the config directly: some checkpoints (LEGAL-BERT) ship a
        # 2-label map that conflicts with a num_labels argument.
        config = AutoConfig.from_pretrained(checkpoint)
        config.id2label = dict(enumerate(labels))
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.problem_type = "multi_label_classification"
        # A fresh classification head is expected here; don't print the missing-weights report.
        previous = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            # A checkpoint that already has a classification head of another size gets a fresh
            # head; every other weight still has to match.
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, config=config, ignore_mismatched_sizes=True
            )
        finally:
            hf_logging.set_verbosity(previous)
        return cls(labels, tokenizer, model, max_length, device)

    def _encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.inference_mode()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, len(self.labels)))
        self.model.eval()
        order = np.argsort([len(t) for t in texts])  # similar lengths -> less padding
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            logits = self.model(**self._encode([texts[i] for i in idx])).logits
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        return out

    def save(self, path: Path) -> None:
        self._write_kind(path)
        self.model.save_pretrained(path / "hf")
        self.tokenizer.save_pretrained(path / "hf")
        (path / "transformer.json").write_text(json.dumps({"max_length": self.max_length}))

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> TransformerModel:
        meta = json.loads((path / "kind.json").read_text())
        max_length = json.loads((path / "transformer.json").read_text())["max_length"]
        tokenizer = AutoTokenizer.from_pretrained(path / "hf")
        model = AutoModelForSequenceClassification.from_pretrained(path / "hf")
        return cls(meta["labels"], tokenizer, model, max_length, pick_device(device))


def _linear_schedule(warmup: int, total: int) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return factor


def train_transformer(
    spec: ModelSpec,
    labels: list[str],
    texts: Sequence[str],
    y: np.ndarray,
    val_texts: Sequence[str],
    val_y: np.ndarray,
    *,
    max_length: int,
    pos_weight: PosWeightSpec,
    seed: int,
    out_dir: Path,
    device: str = "auto",
    epochs: int | None = None,
    batch_size: int | None = None,
    log: Callable[[str], None] = print,
) -> tuple[TransformerModel, dict]:
    set_seed(seed)
    dev = pick_device(device)
    epochs = epochs or spec.epochs
    batch_size = batch_size or spec.batch_size
    assert spec.checkpoint is not None
    wrapper = TransformerModel.from_checkpoint(spec.checkpoint, labels, max_length, dev)
    tokenizer, model = wrapper.tokenizer, wrapper.model

    encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
    features = [{k: encoded[k][i] for k in encoded} for i in range(len(texts))]
    targets = torch.tensor(y, dtype=torch.float32)

    def collate(indices: list[int]):
        batch = tokenizer.pad([features[i] for i in indices], return_tensors="pt")
        return {k: v.to(dev) for k, v in batch.items()}, targets[indices].to(dev)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(range(len(texts))),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": spec.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=spec.lr)
    steps_per_epoch = math.ceil(len(loader) / spec.grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _linear_schedule(int(spec.warmup_ratio * total_steps), total_steps)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights(y, pos_weight).to(dev))

    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    started = time.time()
    model_dir = out_dir / "model"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start, running, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, batch_y) in enumerate(loader, start=1):
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(**inputs).logits
            loss = loss_fn(logits.float(), batch_y)
            scaler.scale(loss / spec.grad_accum).backward()
            if step % spec.grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            n_batches += 1

        val_p = wrapper.predict_proba(val_texts)
        score = macro_pr_auc(val_y, val_p)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, n_batches),
            "val_macro_pr_auc": score,
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(record)
        improved = score > best_score
        log(
            f"epoch {epoch}/{epochs}  loss {record['train_loss']:.4f}  "
            f"val macro PR-AUC {score:.4f}  ({record['seconds']}s){'  *best*' if improved else ''}"
        )
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            wrapper.save(model_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= spec.patience:
                log(f"early stopping: no improvement for {spec.patience} epochs")
                break

    best = TransformerModel.load(model_dir, device=str(dev))
    info = {
        "best_epoch": best_epoch,
        "best_val_macro_pr_auc": best_score,
        "history": history,
        "device": str(dev),
        "amp": str(amp_dtype) if use_amp else "off",
        "train_seconds": round(time.time() - started, 1),
    }
    return best, info
