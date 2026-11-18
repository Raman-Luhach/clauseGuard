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
MAX_COMPARE_UNITS = 4000
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
        options.log_severity_level = 3  # errors only: INT8 graphs log harmless folding warnings
        session = ort.InferenceSession(
            str(path / meta["files"][variant]), options, providers=["CPUExecutionProvider"]
        )
        return cls(meta["labels"], session, tokenizer, meta["input_names"], variant)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 4) -> np.ndarray:
        """Probabilities per text. Texts are sorted by token count and run in small batches,
        each cut to its own longest text: on a CPU the work is proportional to the padded
        tokens, and small, evenly sized batches waste almost none (measured: 2-3x faster
        than batches of 32 sorted by characters, identical outputs)."""
        if not len(texts):
            return np.zeros((0, len(self.labels)))
        encodings = self.tokenizer.encode_batch(list(texts))  # right-padded to the longest
        lengths = np.array([sum(e.attention_mask) for e in encodings])
        order = np.argsort(lengths, kind="stable")
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            width = int(lengths[idx].max())
            arrays = {
                "input_ids": [encodings[i].ids[:width] for i in idx],
                "attention_mask": [encodings[i].attention_mask[:width] for i in idx],
                "token_type_ids": [encodings[i].type_ids[:width] for i in idx],
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


def quantize_int8(fp32: Path, int8: Path) -> None:
    """Dynamic INT8 quantisation that is safe on every CPU.

    Plain 8-bit signed weights saturate on x86 CPUs without VNNI instructions (most cloud
    machines, including Kaggle's and Hugging Face's), which destroyed a fine-tuned LEGAL-BERT
    (macro-F1 0.0). 7-bit weights (reduce_range) quantised per channel avoid the overflow, as
    ONNX Runtime's documentation recommends; compare_backends still checks the result.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True
    )


def optimize_graph(src: Path, dst: Path, log: Callable[[str], None] = print) -> bool:
    """ONNX Runtime's transformer optimizer: fuses LayerNorm + residual (SkipLayerNorm),
    bias + GELU and, where it recognises the pattern, attention. Same weights, same precision,
    fewer operations on the CPU. Returns False (and writes nothing) when it fails."""
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        model = optimize_model(str(src), model_type="bert", num_heads=0, hidden_size=0, opt_level=1)
        model.save_model_to_file(str(dst))
    except Exception as exc:  # unknown architecture pattern, old onnxruntime, ...
        log(f"graph optimisation skipped: {exc}")
        return False
    fused = {k: v for k, v in model.get_fused_operator_statistics().items() if v}
    log(f"optimised graph -> {dst.name} (fused: {fused})")
    return True


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
    # The exporter drops inputs the model ignores (DeBERTa-v3 has no token types): feed only
    # what the graph declares.
    import onnx

    graph_inputs = {
        i.name for i in onnx.load(str(out / "model.onnx"), load_external_data=False).graph.input
    }
    names = [name for name in names if name in graph_inputs]
    files = {"fp32": "model.onnx"}
    if optimize_graph(out / "model.onnx", out / "model.opt.onnx", log=log):
        files["fp32-opt"] = "model.opt.onnx"
    if quantize and exporter == "torchscript":
        quantize_int8(out / "model.onnx", out / "model.int8.onnx")
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
    """Macro-F1 and speed of PyTorch vs ONNX variants on the same units; picks what to serve:
    the fastest variant within ``MAX_F1_DROP`` of the reference.

    At most ``MAX_COMPARE_UNITS`` units are used (a fixed random sample): enough to measure
    the F1 change, without running three models over 14k contract sentences on a CPU."""
    from clauseguard.evaluation.metrics import compute_metrics
    from clauseguard.models.transformer import TransformerModel

    if len(texts) > MAX_COMPARE_UNITS:
        keep = np.sort(
            np.random.default_rng(0).choice(len(texts), MAX_COMPARE_UNITS, replace=False)
        )
        texts, y = [texts[i] for i in keep], y[keep]

    # The reference is PyTorch when the run still has its weights, else the fp32 ONNX graph
    # (runs copied back from a GPU machine keep only ONNX).
    candidates: dict[str, Callable[[], MultiLabelModel]] = {}
    if (run_path / "model" / "kind.json").exists():
        candidates["torch"] = lambda: TransformerModel.load(run_path / "model", device="cpu")
    meta = json.loads((run_path / EXPORT_DIR / "onnx.json").read_text())
    for variant in meta["files"]:
        candidates[f"onnx-{variant}"] = lambda v=variant: OnnxModel.load(run_path / EXPORT_DIR, v)
    results, reference = {}, None
    for name, factory in candidates.items():
        try:
            model = factory()
            started = time.perf_counter()  # prediction time only, not loading
            p = model.predict_proba(texts)
        except Exception as exc:  # e.g. a fused graph this onnxruntime can't run: skip it
            if name in ("torch", "onnx-fp32"):
                raise
            log(f"{name:10s} skipped: {exc}")
            continue
        seconds = time.perf_counter() - started
        metrics = compute_metrics(y, p, thresholds, label_ids)
        reference = p if reference is None else reference
        results[name] = {
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro"]["f1"],
            "ms_per_unit": round(1000 * seconds / max(len(texts), 1), 2),
            "max_prob_diff_vs_reference": float(np.abs(p - reference).max()) if len(p) else 0.0,
        }
        log(
            f"{name:10s} macro-F1 {metrics['macro_f1']:.4f}  {results[name]['ms_per_unit']} ms/unit"
        )
    # Serve the fastest variant that costs at most MAX_F1_DROP macro-F1 (fp32 always qualifies).
    reference_f1 = next(iter(results.values()))["macro_f1"]
    allowed = [
        name.removeprefix("onnx-")
        for name, r in results.items()
        if name.startswith("onnx-")
        and (name == "onnx-fp32" or reference_f1 - r["macro_f1"] <= MAX_F1_DROP)
    ]
    serve = min(allowed, key=lambda v: results[f"onnx-{v}"]["ms_per_unit"])
    meta["serve"] = serve
    (run_path / EXPORT_DIR / "onnx.json").write_text(json.dumps(meta, indent=2))
    report = {"n_units": len(texts), "results": results, "serve": serve, "max_f1_drop": MAX_F1_DROP}
    (run_path / EXPORT_DIR / "export_report.json").write_text(json.dumps(report, indent=2))
    log(f"serving onnx-{serve}")
    return report
