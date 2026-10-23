"""The interface every classifier implements, and loading a saved one."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clauseguard.errors import ModelNotAvailableError


class MultiLabelModel(ABC):
    kind: str
    labels: list[str]

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), len(labels)) with probabilities."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    def _write_kind(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind.json").write_text(json.dumps({"kind": self.kind, "labels": self.labels}))


def serving_model_dir(run_path: Path, backend: str | None = None) -> Path:
    """Where to load a run's model from: its ONNX export when there is one (``auto``), or the
    original model. Set ``CLAUSEGUARD_BACKEND`` to ``torch`` or ``onnx`` to force one."""
    backend = backend or os.environ.get("CLAUSEGUARD_BACKEND", "auto")
    if backend not in ("auto", "torch", "onnx"):
        raise ValueError(f"CLAUSEGUARD_BACKEND must be auto, torch or onnx, not {backend!r}")
    has_onnx = (run_path / "onnx" / "onnx.json").exists()
    if backend == "onnx" and not has_onnx:
        raise ModelNotAvailableError(
            f"{run_path.name} has no ONNX export; run `clauseguard export-onnx` first"
        )
    return run_path / "onnx" if backend != "torch" and has_onnx else run_path / "model"


def load_model(path: Path, device: str = "auto") -> MultiLabelModel:
    kind = json.loads((path / "kind.json").read_text())["kind"]
    if kind == "majority":
        from clauseguard.models.baselines import MajorityModel

        return MajorityModel.load(path)
    if kind == "tfidf":
        from clauseguard.models.baselines import TfidfModel

        return TfidfModel.load(path)
    if kind == "onnx":
        from clauseguard.models.onnx import OnnxModel

        return OnnxModel.load(path)
    if kind == "transformer":
        try:
            from clauseguard.models.transformer import TransformerModel
        except ImportError as exc:  # the serving image has no torch
            raise ModelNotAvailableError(
                f"{path.parent.name} needs PyTorch; export it with `clauseguard export-onnx`"
            ) from exc
        return TransformerModel.load(path, device=device)
    raise ValueError(f"unknown model kind {kind!r} in {path}")
