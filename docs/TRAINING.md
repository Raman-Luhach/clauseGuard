# Training the transformer models on a free GPU (Kaggle)

Everything before this step runs on the Mac and is checked by `make check`. This step trains
the transformer models, evaluates them on the test split once, exports them to ONNX and
packages them. Only small results files come back to the Mac.

It runs on **Kaggle**: free, about 30 GPU hours a week, no credit card. Our training needs
roughly 3–5 GPU hours, split into two runs so each stays well inside the session time limit.
(Lightning AI was the original plan; after its October 2026 security incident it revoked all
user access.)

## One-time setup

1. **Kaggle account:** sign up at https://www.kaggle.com, then verify your phone number
   (Settings → Phone verification). Without it, GPUs and internet access are switched off.
2. **GitHub token** (the repository is private): GitHub → Settings → Developer settings →
   Personal access tokens → **Fine-grained tokens** → Generate new token.
   - Repository access: **Only select repositories** → `clauseGuard`.
   - Permissions → Repository permissions → **Contents: Read-only**.
   - Copy the token (it starts with `github_pat_`).
3. **Notebook:** Kaggle → Create → New Notebook → File → **Import Notebook** → upload
   `notebooks/train_on_gpu.ipynb` from this repository.
4. In the notebook's right-hand panel:
   - **Session options → Accelerator: GPU T4 x2**. Avoid the P100: recent PyTorch builds may
     not support it.
   - **Internet: on**.
   - **Add-ons → Secrets → Add secret:** label `GITHUB_TOKEN`, value = the token. Make sure it
     is ticked (attached) for this notebook.

## Run 1: Terms of Service + Privacy (about 1–2 hours)

1. In the first code cell, keep `MODULES = "tos privacy"`.
2. Click **Save Version → Save & Run All (Commit) → Save**. The run continues in the
   background; you can close the browser.
3. Follow progress under the version's **Logs**. Each step prints a `=== hh:mm:ss` header:
   installing, datasets, quick tests, the session in miniature, training, then test results,
   ONNX export, robustness, latency and packaging.
4. When the version shows "Successfully ran", open it → **Output** → download
   `clauseguard-results-tos-privacy.tgz`.

## Run 2: Contracts (about 1–3 hours)

Edit the first cell to `MODULES = "contract"`, then **Save Version → Save & Run All** again.
Download `clauseguard-results-contract.tgz`.

## Back on the Mac

```bash
cd ~/nlp/clauseguard
tar xzf ~/Downloads/clauseguard-results-tos-privacy.tgz
tar xzf ~/Downloads/clauseguard-results-contract.tgz
```

You now have:
- `deploy/models/`: the served models for all three modules (ONNX for transformers);
- `reports/generated/final_results_<modules>.md`: validation and **test** results;
- `reports/generated/*_robustness.md` and `*_benchmark.md`: robustness and latency;
- `experiments/log_<modules>.csv`: every training run.

Then build and run the app (`docs/DEPLOY.md`):

```bash
docker build -t clauseguard . && docker run --rm -p 7860:7860 clauseguard   # http://localhost:7860
```

## If something goes wrong

- **"GitHub token found: False"** or **"git clone failed":** the `GITHUB_TOKEN` secret isn't
  attached to this notebook, or the token lacks access to `clauseGuard`.
- **"no CUDA GPU visible":** the accelerator is off, or set to a GPU this PyTorch can't use.
  Choose **GPU T4 x2** and run again.
- **A run hits the time limit or stops:** run it again. Finished training runs aren't
  repeated within a run; across runs, each starts fresh (that's why the work is split in two).
- The installer automatically switches to a PyTorch build for older GPU drivers when needed
  (you'll see "installing the CUDA 12.6 build instead" in the log).

## Other machines

The notebook calls `scripts/train_on_gpu.sh`, which works on any Linux machine with an NVIDIA
GPU (a university server, a rented cloud GPU):

```bash
git clone https://github.com/Raman-Luhach/clauseGuard.git clauseguard && cd clauseguard
MODULES="tos privacy contract" OUT=. bash scripts/train_on_gpu.sh
```

**Google Colab (free)** also works with the same notebook: Runtime → Change runtime type →
**T4 GPU**; add `GITHUB_TOKEN` under the key icon (Secrets) on the left; Runtime → Run all.
Keep the tab open, because the free tier stops after about 90 minutes without interaction and has
no background execution. The results file appears under `/content` in the Files panel.
