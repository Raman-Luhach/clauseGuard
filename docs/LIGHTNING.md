# Training on Lightning AI

Everything before this step runs on the Mac and is checked by `make check`. This session
trains the transformer models for all three modules, evaluates them on the test split once,
and exports the served models to ONNX, so only small files have to come back to the Mac.

## 1. Start a Studio

- GPU: **L4 (24 GB)**. An A10G is equally good; a T4 works at roughly half the speed.
- Disk: allow about 20 GB (CUDA PyTorch about 6 GB, datasets 0.3 GB, about 13 transformer
  runs of 0.3–0.5 GB each, Hugging Face cache about 1.5 GB).

## 2. Set up (about 5 minutes)

```bash
git clone <your repository URL> clauseguard && cd clauseguard
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }
uv sync                                    # Python 3.12 + all dependencies, CUDA PyTorch on Linux
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
make data                                  # download (checksums verified) + prepare all datasets
make test-fast                             # about 10 seconds
make train-gpu-smoke                       # the whole session in miniature, about 5 minutes
```

`make data` reuses the committed split files in `data/splits/`, so the splits are identical to
the Mac's. If `make train-gpu-smoke` fails, stop and fix it before starting the real session.

## 3. Train (about 5–8 hours on an L4)

Run it in the background so closing the browser tab doesn't stop it:

```bash
nohup make train-gpu > train.log 2>&1 &
tail -f train.log                          # Ctrl-C stops watching, not training
```

What it does, per module (`clauseguard train-gpu`):
1. trains every transformer in `configs/<module>.yaml` with the first seed;
2. picks the best one on validation macro-F1;
3. trains it again with the two other seeds;
4. selects the best run for serving (or the TF-IDF baseline, if that is better).

If the Studio stops, run `make train-gpu` again: finished runs are skipped, unfinished ones
are cleaned up and redone. Progress is in `train.log`, `experiments/log.csv` and
`reports/generated/gpu_session.json`.

## 4. Evaluate, export, package (about 15 minutes)

```bash
make after-gpu
```

This runs, in order: `evaluate-final` (test split, once: `reports/generated/final_results.md`),
`export-onnx` (ONNX + INT8, keeping INT8 only if it loses at most 0.01 macro-F1),
`robustness`, `benchmark`, and `package` (served models into `deploy/models`).

## 5. Bring back only what is needed, then stop the GPU

```bash
tar czf clauseguard-results.tgz deploy/models reports/generated experiments data/data_cards
```

Download `clauseguard-results.tgz` from the Studio's file browser (a few hundred MB), then
**stop the Studio**. On the Mac, inside `clauseguard/`:

```bash
tar xzf ~/Downloads/clauseguard-results.tgz
docker build -t clauseguard . && docker run -p 7860:7860 clauseguard   # http://localhost:7860
```

The full PyTorch checkpoints stay on Lightning. Download `artifacts/runs` too only if you
want to re-export or re-evaluate on the Mac (about 5 GB).
