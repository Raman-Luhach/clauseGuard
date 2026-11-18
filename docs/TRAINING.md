# Training the transformer models on a GPU

Everything before this step runs on the Mac and is checked by `make check`. This step trains
the transformer models, evaluates them on the test split, exports them to ONNX, trains the
joint model, and writes **one results archive**. Only that archive comes back to the Mac;
PyTorch weights stay on the GPU machine.

Two sessions have been run so far:

1. **October 2026, Kaggle (free T4):** DistilBERT, LEGAL-BERT and PrivBERT for all three
   modules. These models are served today.
2. **Second session, Lambda (this guide):** the improvements decided after the error analysis
   (`docs/ERROR_ANALYSIS.md`): rebuilt contract sentences, oversampling of rare labels,
   recall-targeted thresholds, DeBERTa-v3 challengers, more contract training sentences, and
   the joint model. What each module trains is listed under `gpu_candidates` in
   `configs/<module>.yaml`:

| Module | Candidates | Then |
|---|---|---|
| ToS | LEGAL-BERT + oversampling, DeBERTa-v3, DeBERTa-v3 + oversampling | best one with 2 more seeds |
| Privacy | PrivBERT + oversampling, DeBERTa-v3, DeBERTa-v3 + oversampling | best one with 2 more seeds |
| Contract | LEGAL-BERT and DeBERTa-v3, each with and without oversampling, all with ~44k training sentences | best one with 2 more seeds |
| Joint | one LEGAL-BERT over all three modules' sentences | test once, report |

## Lambda: step by step

### 1. Launch an instance (about 2 minutes)

1. Sign in at https://cloud.lambda.ai → **Instances → Launch instance**.
2. Instance type: **1x H100 (80 GB PCIe)**. A **1x A100 (40 GB)** also works and costs about
   half as much per hour, but takes roughly twice as long, so the total is similar. Don't pick
   **GH200**: its ARM processor can break some Python packages. Don't pick 8x anything: the
   trainer uses one GPU.
3. Region: any with capacity. Base image: the default **Lambda Stack** (Ubuntu with CUDA).
   Filesystem: none needed.
4. SSH key: add your Mac's public key (`cat ~/.ssh/id_ed25519.pub`; create one with
   `ssh-keygen -t ed25519` if you have none).
5. Launch, wait until it shows **Running**, and copy its IP address.

### 2. Copy the code from the Mac (about 1 minute)

```bash
cd ~/nlp/clauseguard
IP=203.0.113.10   # your instance's IP
rsync -az --exclude .venv --exclude data/raw --exclude data/processed --exclude artifacts \
  --exclude deploy/models --exclude reports/generated --exclude logs --exclude '*.tgz' \
  ./ ubuntu@$IP:~/clauseguard/
```

(Alternatively push to GitHub and `git clone` on the instance; the repository is private, so
that needs a token. Copying with rsync avoids that.)

### 3. Start the session (one command)

```bash
ssh ubuntu@$IP
tmux new -s cg                         # keeps running if your connection drops
cd ~/clauseguard && OUT=~ bash scripts/train_on_gpu.sh 2>&1 | tee ~/train.log
```

Detach with **Ctrl-b, then d**; you can close the terminal and the Mac. To look again:
`ssh ubuntu@$IP` then `tmux attach -t cg` (or `tail -f ~/train.log`).

Each step prints a `=== hh:mm:ss` header. Expected durations on an H100:

| Step | Time |
|---|---|
| Install (Python 3.12, PyTorch), datasets, quick tests, session in miniature | 15–25 min |
| ToS: 3 candidates, then 2 more seeds of the best | ~25 min |
| Privacy: 3 candidates, then 2 more seeds | ~35 min |
| Contract: 4 candidates, then 2 more seeds | ~2–2.5 h |
| Test results, ONNX export, robustness, latency, packaging | ~30 min |
| Joint model | ~40 min |
| **Total** | **about 4.5–5 h (≈ $12 at ~$2.50/h)** |

Measured on 18 November 2026 on a **1x A100 (40 GB)**: 3 h 15 min in total (install to archive),
of which training 1 h 40 min, final evaluation and packaging 1 h 05 min, joint model 8 min.
Copying the 1.5 GB archive back took about 7 minutes.

If the session stops (error, lost instance), run the same command again: finished training
runs are recognised and skipped.

### 4. Copy the results back and stop paying (2 minutes)

When the log ends with `=== ... done: /home/ubuntu/clauseguard-results-tos-privacy-contract.tgz`:

```bash
# on the Mac
scp ubuntu@$IP:~/clauseguard-results-tos-privacy-contract.tgz ~/Downloads/
```

Then **terminate the instance** in the Lambda dashboard (Instances → ⋯ → Terminate). Lambda
bills by the minute until it is terminated, even when the GPU is idle.

### 5. On the Mac: import and refresh (about 1 hour, unattended)

```bash
cd ~/nlp/clauseguard
uv run clauseguard results import ~/Downloads/clauseguard-results-tos-privacy-contract.tgz
make after-import
```

`results import` adds every new run next to the existing ones, keeps the GPU machine's reports
in `reports/generated/gpu-<date>/`, and then, per module, serves whichever is better on
validation: the model served so far or the session's best. (If the served model was scored on
an older version of the data, it is scored again first, so both are compared on the same
units.) `make after-import` rewrites the results table, sets up TF-IDF screening for the
served ToS and contract transformers (the transformer then reads only the units the fast model
gives some chance of a label: 2-3x faster, same flags), reruns robustness and latency on the
Mac, packages the served models into `deploy/models/`, and writes the joint-model report.

## Kaggle (free, slower): the first session's route

The notebook `notebooks/train_on_gpu.ipynb` runs the same script on Kaggle's free T4 GPUs.
One-time setup:

1. **Kaggle account** with a verified phone number (Settings → Phone verification);
   otherwise GPUs and internet access are off.
2. **GitHub token** (the repository is private): GitHub → Settings → Developer settings →
   Personal access tokens → **Fine-grained tokens** → only `clauseGuard`, **Contents:
   Read-only**.
3. Kaggle → Create → New Notebook → File → **Import Notebook** → `notebooks/train_on_gpu.ipynb`.
4. In the right-hand panel: **Accelerator: GPU T4 x2**, **Internet: on**, and under
   **Add-ons → Secrets** a secret `GITHUB_TOKEN` with the token (ticked for this notebook).

Kaggle sessions stop after 12 hours, and a T4 is several times slower than an H100, so split
the work: `MODULES = "tos privacy"` first, then `MODULES = "contract"` (and set `JOINT=0` for
one of them). Each run: **Save Version → Save & Run All (Commit)**, then download the `.tgz`
from the version's **Output** and import it on the Mac as in step 5.

## Other machines

`scripts/train_on_gpu.sh` works on any Linux machine with an NVIDIA GPU:

```bash
OUT=~ bash scripts/train_on_gpu.sh                         # everything
MODULES="contract" JOINT=0 OUT=~ bash scripts/train_on_gpu.sh  # one module, no joint model
```

## If something goes wrong

- **"no CUDA GPU visible":** the machine has no usable GPU, or the driver is too old for the
  PyTorch build. The script switches to the CUDA 12.6 build of PyTorch by itself when needed
  (you'll see "installing the CUDA 12.6 build instead").
- **A download fails** (dataset hosts are occasionally down): run the command again; finished
  downloads are kept. The CLAUDETTE download is optional: without it ToS still trains, but
  documents get no names and document-level ToS metrics are skipped.
- **Out of GPU memory** (only on GPUs smaller than 24 GB): lower `batch_size` for the
  candidates in `configs/contract.yaml` (64 → 32) and run again.
- **The session stopped half-way:** run the command again; finished runs are skipped.
