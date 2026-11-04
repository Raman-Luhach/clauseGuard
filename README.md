# ClauseGuard

ClauseGuard reads terms of service, privacy policies and contracts, flags clauses that could harm the reader, and explains each one in plain language. It is an awareness tool, not legal advice.

The user always picks the document type first. Each type has its own analyzer, trained on its own dataset at that dataset's native level of annotation:

| Document type | `--type` | Training data | Unit classified | Status |
|---|---|---|---|---|
| Terms of Service | `tos` | UNFAIR-ToS (LexGLUE) + CLAUDETTE document names | sentence | ✅ |
| Privacy Policy | `privacy` | OPP-115 (2-of-3 annotator majority, 3 sub-labels) | paragraph | ✅ |
| Contract | `contract` | CUAD v1 (expert spans → sentences, 19 clause types) | sentence | ✅ |

Every module works end to end on the CPU with its TF-IDF baseline. The transformer models
(DistilBERT, LEGAL-BERT, PrivBERT) pass `--smoke` and are trained in one GPU session
on a free Kaggle GPU ([docs/TRAINING.md](docs/TRAINING.md)); the app then serves them through ONNX Runtime.

See [docs/PLAN.md](docs/PLAN.md) for the full plan.

## Setup

```bash
uv sync          # installs Python 3.12 and every dependency (including torch)
make data        # downloads the datasets (SHA-256 verified) and builds unit tables + data cards
make train-cpu   # trains the majority and TF-IDF baselines for all modules (~2 minutes)
make evaluate    # validation reports (per label, document level, clause recall)
make cv          # 5-fold cross-validation of the privacy baseline
make example     # analyses one sample document per type
make check       # lint + tests + smoke runs + GPU session in miniature: run before the GPU
make serve       # web UI + API on http://127.0.0.1:8000 (API docs at /docs)
```

## Web app and API

`make serve` starts the web UI and REST API. The user picks the document type, uploads a PDF or
text file (or pastes text) and gets the highlighted document, the flagged clauses with
explanations, the risk score and, for privacy policies, the protections found and missing.

```bash
curl -F document_type=contract -F file=@agreement.pdf http://127.0.0.1:8000/analyze
curl -F document_type=tos -F "text=<terms.txt" http://127.0.0.1:8000/analyze
```

| Endpoint | Purpose |
|---|---|
| `GET /` | web UI |
| `GET /health` | which modules have a model (503 if none) |
| `GET /document-types` | the types to choose from |
| `POST /analyze` | form fields `document_type` (required) and `file` or `text` |
| `POST /feedback` | `{document_type, model_run, label, agree}`; no document text is stored |

Errors return `{"error": {"code", "message"}}`: 422 for unusable input (no type, empty, too
short, scanned PDF, not English), 413 for files over 20 MB, 503 when the module has no model.

**Explanations** come from templates. Set `CLAUSEGUARD_LLM=anthropic` (and an Anthropic API key)
to have Claude write them instead; only flagged clauses are sent, each answer is checked, and
any failure falls back to the template. Settings are in `src/clauseguard/explain/llm.py`.

**Monitoring:** every request is logged as metadata only (`logs/events.jsonl`, never the text);
`clauseguard monitor` summarises volume, errors, latency and feedback, and flags labels firing
much more or less often than on validation data.

## After the GPU session

```bash
make after-gpu   # test results, ONNX export (INT8 if it costs <= 0.01 macro-F1),
                 # robustness, latency benchmark, packaged models in deploy/models
```

See [docs/TRAINING.md](docs/TRAINING.md) for the session itself (a ready-made Kaggle notebook,
`notebooks/train_on_gpu.ipynb`, runs `scripts/train_on_gpu.sh`) and
[docs/DEPLOY.md](docs/DEPLOY.md) for Docker and Hugging Face Spaces. The serving image has no
PyTorch: transformers run through ONNX Runtime.

## Commands

```bash
clauseguard data download all|tos|privacy|contract   # fetch + verify + extract
clauseguard data prepare tos                         # -> data/processed/tos/units.parquet
clauseguard data card tos                            # -> data/data_cards/tos.md
clauseguard train tos --model tfidf [--seed 42] [--smoke]
clauseguard smoke tos privacy contract               # tiny CPU run of every configured model
clauseguard cv privacy [--folds 5]                   # grouped cross-validation (TF-IDF)
clauseguard evaluate tos [--run ID] [--split test --final]
clauseguard models list tos | models select tos RUN_ID
clauseguard analyze FILE.pdf|FILE.txt --type tos [--json] [--out report.json]
clauseguard train-gpu tos privacy contract [--smoke]  # the GPU session (resumable)
clauseguard final-report tos privacy contract [--final]
clauseguard export-onnx tos | robustness tos | benchmark tos | llm-compare tos
clauseguard package-models deploy/models | serve | monitor [--hours 24]
```

Prefix commands with `uv run` (or activate `.venv`). Set `CLAUSEGUARD_DEBUG=1` to see full tracebacks.

- **Models:** `majority` and `tfidf` are trained on the CPU. `distilbert` and `legal-bert` (all modules) and `privbert` (privacy) are trained on a GPU later; `--smoke` checks them on the CPU.
- **The test split** can only be evaluated with `--final`. Model and threshold choices use validation data only.
- **Which run is served:** the one chosen with `models select`, otherwise the best by validation macro-F1 (never the majority baseline).

## Output

`analyze` returns an `AnalysisResult` (see `src/clauseguard/schema.py`). It contains:

- the document type and the model run used;
- a 0–100 risk score and band;
- for privacy policies, the protections found and missing (choices, access/deletion, security, retention);
- one finding per flagged clause, with its text, character offsets into the cleaned text, page (for PDFs), labels with confidence, risk group, severity and a plain-language explanation;
- warnings and a disclaimer.

## Layout

```
configs/            tos.yaml, privacy.yaml, contract.yaml (labels, models, modifiers, data settings)
                    taxonomy.yaml (risk groups, severities, definitions, explanation templates)
src/clauseguard/
  ingestion/        PDF extraction, cleaning (offset-preserving), sentence segmentation, language check
  data/             download manifest + downloader, dataset converters, unit tables, data cards
  models/           majority / TF-IDF / transformer, thresholds, training, run registry
  evaluation/       metrics, error analysis, reports
  scoring/          document risk score, severity modifiers and expected statements
  explain/          template explanations, optional Claude explanations
  modules/          analyzer: segment -> classify -> findings -> score
  pipeline.py       analyze(document_type, text|path|pdf_bytes) with a result cache
  web/              FastAPI app + static web UI (no build step)
  gpu.py            the GPU training session
  deploy.py         package served models for the Docker image
  monitoring.py     metadata-only request log, drift summary
  cli.py
configs/robustness/ hand-written paraphrase, negation and neutral cases per module
docs/               TRAINING.md (GPU session on Kaggle), DEPLOY.md (Docker, Hugging Face Spaces)
notebooks/          train_on_gpu.ipynb (the Kaggle/Colab notebook)
scripts/            train_on_gpu.sh (the whole GPU job on any Linux GPU machine)
data/splits/        document-level split assignments (committed, reused on every rebuild)
data/data_cards/    generated data cards + samples for the manual conversion check (committed)
experiments/log.csv one row per training run (committed)
artifacts/          trained runs (not committed)
examples/           sample documents: GitHub Terms of Service and Privacy Statement (CC0),
                    a CUAD test-split contract (CC BY 4.0)
tests/
```

## Dataset licences

UNFAIR-ToS and CUAD are CC BY 4.0. OPP-115 is licensed for research, teaching and scholarship only. CLAUDETTE is used for research. Raw data is downloaded, never committed.
