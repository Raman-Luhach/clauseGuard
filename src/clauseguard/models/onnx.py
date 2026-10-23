"""Serve fine-tuned transformers with ONNX Runtime instead of PyTorch.

``export_onnx`` writes ``<run>/onnx/`` with a full-precision model, a dynamically quantised
INT8 copy (about 4x smaller and faster on CPU), and the fast tokenizer. ``compare_backends``
measures what each variant costs in macro-F1 against PyTorch and picks the one to serve: INT8
unless it loses more than ``MAX_F1_DROP``. ``OnnxModel`` needs only onnxruntime and tokenizers,
so the deployed app does not install torch.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from clauseguard.models.base import MultiLabelModel

EXPORT_DIR = "onnx"
MAX_F1_DROP = 0.01
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


class OnnxModel(MultiLabelModel):
    kind = "onnx"

    def __init__(self, labels: list[str], session, tokenizer, input_names: list[str], variant: str):
        self.labels = labels
        self.session = session
        self.tokenizer = tokenizer
        self.input_names = input_names
        self.variant = variant

    @classmethod
    def load(cls, path: Path, variant: str | None = None) -> OnnxModel:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((path / "onnx.json").read_text())
        variant = variant or meta["serve"]
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        tokenizer.enable_truncation(meta["max_length"])
        tokenizer.enable_padding(pad_id=meta["pad_id"], pad_token=meta["pad_token"])
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(path / meta["files"][variant]), options, providers=["CPUExecutionProvider"]
        )
        return cls(meta["labels"], session, tokenizer, meta["input_names"], variant)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, len(self.labels)))
        order = np.argsort([len(t) for t in texts])  # similar lengths -> less padding
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            encodings = self.tokenizer.encode_batch([texts[i] for i in idx])
            arrays = {
                "input_ids": [e.ids for e in encodings],
                "attention_mask": [e.attention_mask for e in encodings],
                "token_type_ids": [e.type_ids for e in encodings],
            }
            feeds = {name: np.asarray(arrays[name], dtype=np.int64) for name in self.input_names}
            logits = self.session.run(None, feeds)[0]
            out[idx] = 1.0 / (1.0 + np.exp(-logits))
        return out

    def save(self, path: Path) -> None:
        raise NotImplementedError("ONNX models are written by export_onnx")


def _export_graph(wrapper, sample: dict, names: list[str], path: Path) -> str:
    import torch

    args = tuple(sample[name] for name in names)
    try:
        # The TorchScript exporter: its graphs quantise cleanly with onnxruntime.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                args,
                str(path),
                input_names=names,
                output_names=["logits"],
                dynamic_axes={
                    **{n: {0: "batch", 1: "sequence"} for n in names},
                    "logits": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
            )
        return "torchscript"
    except Exception:  # removed in a future torch: fall back to the torch.export exporter
        batch, seq = torch.export.Dim("batch", min=1, max=1024), torch.export.Dim("seq", min=2)
        torch.onnx.export(
            wrapper,
            args,
            str(path),
            input_names=names,
            output_names=["logits"],
            dynamic_shapes={n: {0: batch, 1: seq} for n in names},
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        return "dynamo"


def export_onnx(
    run_path: Path, *, quantize: bool = True, log: Callable[[str], None] = print
) -> dict:
    """Export ``<run>/model`` (a fine-tuned transformer) to ``<run>/onnx``."""
    import torch

    from clauseguard.models.transformer import TransformerModel

    model = TransformerModel.load(run_path / "model", device="cpu")
    hf, tokenizer = model.model.eval(), model.tokenizer
    sample = tokenizer(
        ["a slightly longer example sentence used to trace the graph", "short"],
        padding=True,
        return_tensors="pt",
    )
    names = [name for name in INPUT_NAMES if name in sample]

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *inputs):
            return self.inner(**dict(zip(names, inputs, strict=True))).logits

    out = run_path / EXPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    exporter = _export_graph(Logits(hf).eval(), sample, names, out / "model.onnx")
    hf.eval()  # the TorchScript exporter restores the module's previous training flag
    files = {"fp32": "model.onnx"}
    if quantize and exporter == "torchscript":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8
        )
        files["int8"] = "model.int8.onnx"
    backend = getattr(tokenizer, "backend_tokenizer", None) or tokenizer._tokenizer
    backend.save(str(out / "tokenizer.json"))
    meta = {
        "labels": model.labels,
        "max_length": model.max_length,
        "input_names": names,
        "pad_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
        "files": files,
        "serve": "int8" if "int8" in files else "fp32",
        "exporter": exporter,
    }
    (out / "onnx.json").write_text(json.dumps(meta, indent=2))
    (out / "kind.json").write_text(json.dumps({"kind": "onnx", "labels": model.labels}))
    sizes = {k: round((out / f).stat().st_size / 1e6, 1) for k, f in files.items()}
    log(f"exported {run_path.name} with the {exporter} exporter -> {out} (MB: {sizes})")
    return meta


def compare_backends(
    run_path: Path,
    texts: list[str],
    y: np.ndarray,
    thresholds: list[float],
    label_ids: list[str],
    *,
    log: Callable[[str], None] = print,
) -> dict:
    """Macro-F1 and speed of PyTorch vs ONNX variants on the same units; picks what to serve."""
    from clauseguard.evaluation.metrics import compute_metrics
    from clauseguard.models.transformer import TransformerModel

    candidates: dict[str, Callable[[], MultiLabelModel]] = {
        "torch": lambda: TransformerModel.load(run_path / "model", device="cpu"),
    }
    meta = json.loads((run_path / EXPORT_DIR / "onnx.json").read_text())
    for variant in meta["files"]:
        candidates[f"onnx-{variant}"] = lambda v=variant: OnnxModel.load(run_path / EXPORT_DIR, v)
    results, reference = {}, None
    for name, factory in candidates.items():
        model = factory()
        started = time.perf_counter()
        p = model.predict_proba(texts)
        seconds = time.perf_counter() - started
        metrics = compute_metrics(y, p, thresholds, label_ids)
        reference = p if reference is None else reference
        results[name] = {
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro"]["f1"],
            "ms_per_unit": round(1000 * seconds / max(len(texts), 1), 2),
            "max_prob_diff_vs_torch": float(np.abs(p - reference).max()) if len(p) else 0.0,
        }
        log(
            f"{name:10s} macro-F1 {metrics['macro_f1']:.4f}  {results[name]['ms_per_unit']} ms/unit"
        )
    serve = "fp32"
    if "onnx-int8" in results:
        drop = results["torch"]["macro_f1"] - results["onnx-int8"]["macro_f1"]
        serve = "int8" if drop <= MAX_F1_DROP else "fp32"
    meta["serve"] = serve
    (run_path / EXPORT_DIR / "onnx.json").write_text(json.dumps(meta, indent=2))
    report = {"n_units": len(texts), "results": results, "serve": serve, "max_f1_drop": MAX_F1_DROP}
    (run_path / EXPORT_DIR / "export_report.json").write_text(json.dumps(report, indent=2))
    log(f"serving onnx-{serve}")
    return report
