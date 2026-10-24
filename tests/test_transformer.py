"""Exercises the real training loop with a tiny random BERT (downloads ~1 MB once)."""

from __future__ import annotations

import numpy as np
import pytest

from clauseguard.config import ModelSpec, PosWeightSpec
from clauseguard.models.base import load_model
from clauseguard.models.transformer import pos_weights, train_transformer

pytestmark = pytest.mark.slow
TINY = "hf-internal-testing/tiny-random-bert"  # ships tokenizer files transformers v5 needs


@pytest.fixture(scope="module")
def tiny_checkpoint() -> str:
    """Local path of the tiny model; only touches the network the first time."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns = ["*.json", "*.txt", "*.safetensors"]
    try:
        return snapshot_download(TINY, allow_patterns=patterns, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(TINY, allow_patterns=patterns)


def test_pos_weights_modes():
    y = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])
    assert pos_weights(y, PosWeightSpec(mode="linear", max=100)).tolist() == [3.0, 1.0]
    assert pos_weights(y, PosWeightSpec(mode="sqrt", max=100))[0].item() == pytest.approx(3**0.5)
    assert pos_weights(y, PosWeightSpec(mode="linear", max=2))[0].item() == 2.0


def test_train_save_reload(tmp_path, tiny_checkpoint):
    texts = [
        "we may terminate your account at any time",
        "disputes go to binding arbitration",
        "you can contact support by email",
        "we share photos with your friends",
    ] * 8
    y = np.array([[1, 0], [0, 1], [0, 0], [0, 0]] * 8)
    spec = ModelSpec(
        type="transformer", checkpoint=tiny_checkpoint, lr=1e-3, epochs=3, batch_size=8, patience=2
    )
    model, info = train_transformer(
        spec,
        ["a", "b"],
        texts,
        y,
        texts,
        y,
        max_length=32,
        pos_weight=PosWeightSpec(),
        seed=0,
        out_dir=tmp_path,
        device="cpu",
        log=lambda *_: None,
    )
    assert 1 <= info["best_epoch"] <= 3 and len(info["history"]) >= 1
    p = model.predict_proba(texts[:4])
    assert p.shape == (4, 2) and ((p >= 0) & (p <= 1)).all()
    reloaded = load_model(tmp_path / "model", device="cpu")
    np.testing.assert_allclose(reloaded.predict_proba(texts[:4]), p, atol=1e-5)
    assert reloaded.predict_proba([]).shape == (0, 2)
