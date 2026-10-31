"""ONNX export of a fine-tuned transformer, and serving it without PyTorch."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clauseguard.config import ModelSpec, PosWeightSpec
from clauseguard.errors import ModelNotAvailableError
from clauseguard.models.base import load_model, serving_model_dir
from clauseguard.models.onnx import compare_backends, export_onnx
from clauseguard.models.transformer import train_transformer
from tests.test_transformer import tiny_checkpoint  # noqa: F401  (fixture)

pytestmark = pytest.mark.slow
TEXTS = [
    "we may terminate your account at any time",
    "disputes go to binding arbitration",
    "you can contact support by email",
    "we share photos with your friends",
] * 8
Y = np.array([[1, 0], [0, 1], [0, 0], [0, 0]] * 8)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, tiny_checkpoint):  # noqa: F811
    run = tmp_path_factory.mktemp("run")
    spec = ModelSpec(
        type="transformer", checkpoint=tiny_checkpoint, lr=1e-3, epochs=2, batch_size=8, patience=2
    )
    train_transformer(
        spec,
        ["a", "b"],
        TEXTS,
        Y,
        TEXTS,
        Y,
        max_length=16,
        pos_weight=PosWeightSpec(),
        seed=0,
        out_dir=run,
        device="cpu",
        log=lambda *_: None,
    )
    export_onnx(run, log=lambda *_: None)
    return run


def test_onnx_matches_torch(exported):
    torch_model = load_model(exported / "model", device="cpu")
    fp32 = load_model(exported / "onnx")  # serves int8 by default
    long_text = "word " * 200  # longer than max_length: truncation must match too
    texts = [*TEXTS[:4], long_text, "ok"]
    from clauseguard.models.onnx import OnnxModel

    exact = OnnxModel.load(exported / "onnx", "fp32")
    np.testing.assert_allclose(
        exact.predict_proba(texts), torch_model.predict_proba(texts), atol=1e-4
    )
    assert fp32.predict_proba(texts).shape == (6, 2)
    assert fp32.predict_proba([]).shape == (0, 2)


def test_tokenizer_matches_hugging_face(exported):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    meta = json.loads((exported / "onnx" / "onnx.json").read_text())
    fast = Tokenizer.from_file(str(exported / "onnx" / "tokenizer.json"))
    fast.enable_truncation(meta["max_length"])
    hf = AutoTokenizer.from_pretrained(exported / "model" / "hf")
    for text in ["Hello, World!", "word " * 100]:
        expected = hf(text, truncation=True, max_length=meta["max_length"])["input_ids"]
        assert fast.encode(text).ids == expected


def test_compare_picks_a_variant_and_serving_prefers_onnx(exported, monkeypatch):
    report = compare_backends(exported, TEXTS, Y, [0.5, 0.5], ["a", "b"], log=lambda *_: None)
    assert set(report["results"]) == {"torch", "onnx-fp32", "onnx-int8"}
    assert report["results"]["onnx-fp32"]["max_prob_diff_vs_torch"] < 1e-4
    assert report["serve"] in ("int8", "fp32")
    assert serving_model_dir(exported) == exported / "onnx"
    assert serving_model_dir(exported, "torch") == exported / "model"
    monkeypatch.setenv("CLAUSEGUARD_BACKEND", "onnx")
    assert serving_model_dir(exported) == exported / "onnx"


def test_onnx_backend_without_export_is_a_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="export-onnx"):
        serving_model_dir(tmp_path, "onnx")
    with pytest.raises(ValueError, match="CLAUSEGUARD_BACKEND"):
        serving_model_dir(tmp_path, "tensorflow")
