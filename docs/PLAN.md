# ClauseGuard: Build Plan

ClauseGuard reads a legal or policy document, flags clauses that could harm the reader, groups them into risk categories with a severity, gives the document a risk score, and explains each flag in plain language.

> **Status (2026-10-23):** Parts 1–4 are coded and checked on the Mac (`make check`). All three modules work end to end with their TF-IDF baselines, in the CLI, the API and the web UI. All 13 configured models pass `--smoke`, and the GPU session itself passes in miniature. Next: the GPU session (section 6, step by step in `docs/LIGHTNING.md`).

It has three separate analyzers:

| Module | What the user uploads | Training data | How the data is labelled | Unit the model classifies |
|---|---|---|---|---|
| **Terms of Service (ToS)** | App/website terms | UNFAIR-ToS (LexGLUE) | Each **sentence** | Sentence |
| **Privacy Policy** | Website/app privacy policy | OPP-115 | Each **paragraph** (segment), with text **spans** inside it | Paragraph |
| **Contract** | Business agreement | CUAD v1 | Text **spans** (a phrase to a few sentences) | Sentence |

**User flow.** The user picks the document type first, then uploads or pastes the document. Only the matching analyzer runs. There is no model that guesses the document type.

---

## 1. Key design decisions

1. **Each module keeps its own data, label set and model.** We don't force the three datasets into one format yet. Each module is trained on the unit that fits its data best.
2. **All modules return the same output format.** Every flag is a `Finding` with the module, unit type, text, character offsets in the document, labels with confidence, risk group, severity and explanation. Because of this, combining modules later only needs a new mapping layer, not a rewrite.
3. **The core code is shared.** Text extraction, cleaning, segmentation, the training loop, threshold tuning, evaluation, scoring, CLI and API are written once. Modules differ only in their config and data converter.
4. **Data conversion keeps character offsets.** A `TextWithMap` utility records where each character of the cleaned text came from. This lets us move labels between HTML, raw text and clean text without losing their position.
5. **Code first, train last.** All four parts are coded and checked on your Mac. Each training command has a `--smoke` mode (tiny model, a few hundred examples, 1 epoch on CPU), and the TF-IDF models train for real on the Mac, so the whole app works end to end before any GPU is used. Once Part 4 is done, a single Lightning AI session trains every transformer model (see section 6). The app uses the best trained model available for each module: TF-IDF until the transformers exist.
6. **Reproducibility.** Dataset revisions and checksums are pinned, split files are committed, seeds are fixed, and every run is logged.

---

## 2. Data sources (all checked and downloaded successfully on 2026-10-23)

| Dataset | Source | Size | Licence | Notes from inspecting the files |
|---|---|---|---|---|
| UNFAIR-ToS | `huggingface.co/datasets/coastalcph/lex_glue`, folder `unfair_tos/`, parquet files, revision `c23fdff` | 0.9 MB | CC BY 4.0 | 5,532 / 2,275 / 1,607 sentences. Columns are only `text` and `labels` (no document IDs). **Text is lowercased and pre-tokenised** (for example "site , services"), so inference must normalise text the same way. 89% of sentences have no label. Arbitration has only 28 training examples. |
| CLAUDETTE original (optional) | `claudette.eui.eu/ToS.zip` | 2.3 MB | Research | Same 9,414 sentences as UNFAIR-ToS, plus the **50 document names**. Labels are binary per category only (no severity levels). We use it only to add document IDs for checking document-level scores. |
| OPP-115 | `usableprivacy.org/static/data/OPP-115_v1_0.zip` | 99 MB | Research and teaching only (non-commercial) | 115 policies, 3,792 segments, 3 annotators each. **Span offsets index into the raw HTML of each segment** (97.7% exact on the raw HTML, 0% on the tag-stripped text). The rest are found by searching for `selectedText` or are known broken (`documentation/errant_span_indexes/`). Segments: median 59 words, 95th percentile 164, maximum 372, so nothing needs chunking at 512 tokens. |
| CUAD v1 | `github.com/TheAtticusProject/cuad/raw/main/data.zip` | 18 MB | CC BY 4.0 | 510 contracts in SQuAD format. **Official split by contract: 408 train / 102 test, no overlap, no validation set.** All 13,823 span offsets match exactly. Contracts have a median of 33k characters (26.8M in total) and contain EDGAR formatting problems: hard line breaks, runs of blank lines, list markers stuck to words ("a.As of", "Article 1.Amendment"). |
| CUAD full PDFs (optional) | `zenodo.org/records/4595826` → `CUAD_v1.zip` | 106 MB | CC BY 4.0 | The original contract PDFs. Used only to test PDF extraction on real documents. |

The download command checks a SHA-256 for every file, extracts into `data/raw/<name>/`, and does nothing if the files are already present.

---

## 3. Shared architecture

### 3.1 Flow

```
User ─► Step 1: "What kind of document is this?"   (Terms of Service | Privacy Policy | Contract)
     ─► Step 2: upload a PDF or paste text
                    │
                    ▼
         Ingestion: extract text (PDF) → clean (keep offset map) → language and size checks
                    │
       ┌────────────┼─────────────────┐          only the chosen module runs
       ▼            ▼                 ▼
   ToS module   Privacy module    Contract module
   sentences    paragraphs        sentences (adjacent flags merged)
   8 labels     10 categories      19 clause types
                + 3 sub-labels
       └────────────┼─────────────────┘
                    ▼
     Findings (common schema) → taxonomy (risk group, severity) → document risk score
                    ▼
     Explanations (templates, optional LLM) → report (CLI JSON / REST API / web UI)
```

### 3.2 Repository layout

```
clauseguard/
├── pyproject.toml  uv.lock  .python-version (3.12)  Makefile  README.md
├── configs/
│   ├── tos.yaml  privacy.yaml  contract.yaml   # labels, unit type, max_len, model candidates, threshold objective
│   └── taxonomy.yaml                           # label → risk group, severity, definition, template explanation
├── data/                                       # raw/ and processed/ are gitignored
│   ├── raw/  processed/  splits/  data_cards/
├── src/clauseguard/
│   ├── schema.py            # Unit, Finding, AnalysisResult (pydantic)
│   ├── textmap.py           # TextWithMap: strip HTML, clean, unwrap lines, all with character-offset maps
│   ├── ingestion/           # pdf.py, clean.py, segment.py, language.py
│   ├── data/                # download.py, manifest.yaml, tos.py, privacy.py, contract.py, splits.py, card.py
│   ├── models/              # baselines.py, transformer.py, thresholds.py, predictor.py, onnx.py
│   ├── evaluation/          # metrics.py, error_analysis.py, robustness.py
│   ├── scoring/             # taxonomy.py, risk_score.py
│   ├── explain/             # templates.py, llm.py
│   ├── modules/             # tos.py, privacy.py, contract.py: segment → predict → findings
│   ├── pipeline.py          # analyze(document_type, text | pdf) → AnalysisResult
│   ├── monitoring.py
│   └── cli.py
├── api/main.py              # FastAPI
├── src/clauseguard/web/     # FastAPI app + static web UI (HTML/CSS/JS, no build step)
├── artifacts/               # trained models per module/run (gitignored; pushed to HF Hub)
├── experiments/log.csv      # one row per training run
├── tests/                   # unit, converter golden tests, smoke, API, robustness
└── reports/                 # data cards, evaluation reports, final report material
```

### 3.3 Data formats

**Unit table** (one parquet file per module in `data/processed/`). Every module uses these columns, although the unit types differ:

| column | meaning |
|---|---|
| `doc_id` | policy file, contract title, or ToS document (via CLAUDETTE join; otherwise `unknown`) |
| `unit_id` | `{doc_id}:{index}` |
| `unit_type` | `sentence` or `segment` |
| `text` | exactly what the model sees |
| `char_start`, `char_end` | position in the document's clean text (empty for UNFAIR-ToS, which has no full documents) |
| `labels` | list of label names, namespaced (`tos:arbitration`, `privacy:third_party_sharing`, `contract:non_compete`) |
| `split` | `train` / `validation` / `test` (always assigned per document, never per sentence) |
| `meta` | JSON: annotator votes, ambiguity flags, and so on |

**Finding / AnalysisResult** (API and CLI output): `document_type`, `model_version`, `findings[]` (with `text`, offsets, `labels[{label, score}]`, `risk_group`, `severity`, `explanation`, `explanation_source`), `protections_found[]`, `protections_missing[]`, `risk_score` (0–100), `risk_band`, `warnings[]`, `timings_ms`, `disclaimer`.

### 3.4 Segmentation (one implementation, used for both training and inference)

- **Cleaning:** normalise unicode (quotes, dashes, non-breaking spaces), remove repeated PDF headers and footers, join words hyphenated across line breaks, unwrap hard-wrapped lines, separate glued list markers ("a.As" → "a. As"), collapse whitespace. All of it is offset-mapped.
- **Sentences:** split into blocks (blank lines, headings, list items), then into sentences inside each block using spaCy `en_core_web_sm` with only the sentence splitter enabled, plus legal rules. Abbreviations (Inc., Ltd., No., U.S., e.g., i.e.) and references like "Section 4.2(b)" are protected. Fragments under about 20 characters are merged into a neighbour, and units over about 1,500 characters are split at semicolons.
- **Paragraphs** (privacy): blocks, with heading-only blocks merged into the next block, tiny blocks merged, and blocks over about 350 words split at sentence boundaries. The target length matches OPP-115 segments (median 59 words), so inference input looks like the training input.
- **ToS normalisation:** lowercase and put spaces around punctuation so real ToS text looks like the UNFAIR-ToS training text. Covered by a unit test.

### 3.5 Training and evaluation (shared by all modules)

- **Models, in order:** majority baseline → TF-IDF (1–2-grams) + one-vs-rest logistic regression with class weights → DistilBERT → LEGAL-BERT (plus PrivBERT for privacy). All model IDs checked on the Hugging Face Hub: `distilbert/distilbert-base-uncased`, `nlpaueb/legal-bert-base-uncased`, `nlpaueb/legal-bert-small-uncased` (fast CPU option), `mukund/privbert`.
- **Transformer training:** multi-label head, BCE loss with clipped `pos_weight` for imbalance, AdamW, warm-up, fp16 on GPU. Early stopping uses validation **macro PR-AUC**, which doesn't depend on a threshold, so model selection isn't tied to a 0.5 cut-off. Device is chosen automatically (cuda → mps → cpu).
- **Thresholds:** one threshold per label, tuned on validation. The objective can be F1, or F2 when recall should be favoured, which fits the "don't miss harmful clauses" goal.
- **Metrics:** per-label precision, recall, F1 and support; micro and macro F1; macro PR-AUC; "any risky clause" binary precision and recall. Module-specific metrics are listed under each part.
- **Test discipline:** test-set evaluation needs `--final`. Model and threshold choices use validation data only.
- **Logging:** every run writes `artifacts/<module>/<run_id>/` (model, `labels.json`, `thresholds.json`, config, metrics, test predictions) and adds a row to `experiments/log.csv` (git commit, data hash, seed, hyperparameters, metrics, training time).
- **Error analysis:** export false positives and false negatives with text and scores to CSV, plus a table of which labels get predicted together. The CSV is grouped by hand into error types (related labels confused, negation, segmentation error, label noise, ambiguous clause).

### 3.6 CLI

```
clauseguard data download all|tos|privacy|contract   # fetch, verify checksum, extract
clauseguard data prepare  tos|privacy|contract       # convert to unit table + splits
clauseguard data card     tos|privacy|contract       # statistics → data/data_cards/<module>.md
clauseguard train    <module> --model tfidf|distilbert|legal-bert|privbert [--seed N] [--smoke]
clauseguard evaluate <module> --run <id> [--split test --final]
clauseguard analyze  --type tos|privacy|contract <file.pdf|file.txt>
clauseguard serve
```

`make test` runs the unit tests. `make smoke` runs a tiny end-to-end pipeline for all modules on CPU in under 5 minutes.

---

## 4. The four parts

### Part 1: Foundation and the Terms of Service module

UNFAIR-ToS needs no label conversion, so this is the cleanest dataset for building all the shared code. Part 1 ends with a complete, working ToS analyzer.

1. **Scaffold:** uv project (Python 3.12), dependency groups `core`, `train`, `serve`, `dev`; ruff, pytest, Makefile, `.gitignore`, git init.
2. **Download layer for all datasets** (so Parts 2 and 3 can start immediately): `manifest.yaml` with URL, SHA-256 and licence for each file; `clauseguard data download all`.
3. **Shared core:** `schema.py`, `textmap.py`, cleaning, segmentation, PDF extraction (pdfplumber, MIT licence), detection of scanned PDFs with no text layer, language check.
4. **ToS data prep:** parquet → unit table with label names. Optionally join with CLAUDETTE to add document IDs (order-aware text matching; both have 9,414 sentences). Data card: counts per split and label, multi-label rate, token lengths (to confirm `max_len` = 128), duplicate sentences across splits.
5. **Baselines** on your Mac: majority, then TF-IDF + logistic regression.
6. **Generic transformer trainer**, checked on your Mac with `--smoke`. The real DistilBERT and LEGAL-BERT runs happen in the GPU session (section 6).
7. **Threshold tuning, evaluation report, error analysis**, run on the TF-IDF model now and on the transformers after the GPU session. Sanity check for later: LEGAL-BERT should land near the published LexGLUE numbers (about 96 micro-F1 / 83 macro-F1, computed their way with an extra "none" class).
8. **Scoring and output:** `taxonomy.yaml` entries for the 8 ToS labels (table below), generic risk-score code, and `clauseguard analyze --type tos terms.pdf` → full JSON report with template explanations. Each finding also records which page it came from.

| ToS label | Train / test sentences | Risk group | Default severity |
|---|---|---|---|
| Limitation of liability | 191 / 38 | Legal risk | High |
| Unilateral termination | 139 / 38 | Rights & account | High |
| Arbitration | 28 / 7 | Legal risk | High |
| Unilateral change | 122 / 38 | Rights & account | Medium |
| Content removal | 73 / 13 | Rights & account | Medium |
| Jurisdiction | 34 / 16 | Legal risk | Medium |
| Choice of law | 39 / 13 | Legal risk | Low |
| Contract by using | 76 / 23 | Rights & account | Low |

**Done when:** `make test` and `make smoke` pass; the majority and TF-IDF baselines are trained and logged; the transformer trainer passes `--smoke` for every model it will use; the error analysis export works; `analyze --type tos` produces a sensible report on a real terms-of-service PDF using the TF-IDF model.

---

### Part 2: The Privacy Policy module (paragraph level)

OPP-115 categories describe *what a policy does* (collects data, shares data), not whether it is unfair. Three sub-labels from the annotators' attributes make the risk output meaningful, and protective categories feed a "missing protections" check.

1. **Converter** (`data/privacy.py`):
   - Split each `sanitized_policies/*.html` on `|||` into segments. Strip HTML with `TextWithMap`, so raw-HTML span offsets (how 97.7% of spans are stored) map onto clean text.
   - Read the 3 annotators' CSVs. A segment gets a category when **at least 2 of 3 annotators** gave it (majority vote). Don't use span positions from practices listed in `errant_span_indexes/`. If an offset is wrong, find the span by searching for `selectedText`.
   - **Sub-labels** (same majority rule): `negated_practice` (collection or sharing marked "Does Not", for example "we do not sell your data"), `sharing_ads_marketing` (third-party sharing for advertising or marketing), `change_without_personal_notice` (policy changes announced only by posting). `retention_open_ended` has only 35 segments, so it is recorded but not trained.
   - **Derived sentence-level table** using span overlap. It isn't used for the main model. It gives finer highlighting and is ready for the later combination step.
   - **Splits by policy**, 70/15/15, stratified by which labels appear in each policy, seeded and saved to `data/splits/privacy.json`. TF-IDF also gets 5-fold cross-validation by policy, because a test set of about 17 policies is small.
2. **Models:** TF-IDF + logistic regression, DistilBERT, LEGAL-BERT, PrivBERT (trained on privacy policies), `max_len` = 512, 13 outputs (10 categories + 3 sub-labels).
3. **Privacy rules:** sub-labels move the severity of the category they qualify one step on the ladder info < low < medium < high (table below; configured as `modifiers` in `privacy.yaml`). Negation lowers severity by one step rather than hiding the finding, because a paragraph can both share data and say what it doesn't share. Protective categories are reported as positive signals. If no paragraph in a policy covers User Choice/Control, User Access/Edit/Deletion, Data Security or Data Retention, that protection is listed as missing.
4. **Extra metric:** policy-level accuracy of "category present / absent", which is what the missing-protections feature depends on.
5. **Inference:** `analyze --type privacy policy.pdf`. Paragraph segmentation is tuned to OPP-115 segment lengths, and the flagged paragraph is highlighted.

| Privacy output | Segments (majority, all splits) | Risk group | Default severity |
|---|---|---|---|
| Third Party Sharing/Collection | 945 | Personal data | Medium; **High** with `sharing_ads_marketing`; one step lower with `negated_practice` |
| First Party Collection/Use | 1,209 | Personal data | Low; informational with `negated_practice` |
| Data Retention | 78 | Personal data | Medium |
| Policy Change | 119 | Rights & account | Low; **Medium** with `change_without_personal_notice` |
| Do Not Track | 31 | Personal data | Low |
| International & Specific Audiences | 301 | Informational | — |
| User Choice/Control | 358 | Protective | positive signal |
| User Access, Edit & Deletion | 149 | Protective | positive signal |
| Data Security | 210 | Protective | positive signal |
| Other | 1,058 | trained but not shown | — |
| sub-label `sharing_ads_marketing` | 265 | modifier | |
| sub-label `negated_practice` | 160 | modifier | |
| sub-label `change_without_personal_notice` | 108 | modifier | |

**Done when:** converter golden tests pass (a hand-made HTML segment with known spans maps to the right sentences); a manual check of 50 converted sentences is recorded in the data card; TF-IDF is trained and logged, and DistilBERT, LEGAL-BERT and PrivBERT pass `--smoke`; `analyze --type privacy` produces a report with risky practices, protections found and missing protections.

---

### Part 3: The Contract module (spans → sentences)

1. **Converter** (`data/contract.py`):
   - Clean each contract with `TextWithMap` (unwrap EDGAR line breaks, separate glued list markers) so the CUAD offsets, which all match exactly, map onto the clean text.
   - Split into sentences with the shared splitter.
   - **Labelling rule:** sentence *s* gets category *c* when `overlap(s, span) ≥ 0.5 × min(len(s), len(span))`. This covers a short span inside a long sentence and a long span covering several sentences, and it ignores spans that spill a few characters past a sentence boundary. Sentences with only a small overlap are marked ambiguous and left out of training. Validation and test keep every sentence.
   - **Splits:** official 408 train / 102 test by contract. About 41 training contracts (10%) are held out as validation, seeded and stratified.
   - **Negative sampling, training data only:** all positive sentences plus 4 negatives per positive (configurable). Validation and test are never sampled.
   - **Checks:** at least 98% of gold spans must be covered by at least one positive sentence; a manual check of 100 converted sentences goes in the data card.
2. **Category set**, configurable in `contract.yaml`. Rule: risk-relevant and present in at least 50 contracts. Excluded: metadata (document name, parties, dates), descriptive licence types, and very rare categories. The list is finalised after the data card.
3. **Models:** TF-IDF + logistic regression, DistilBERT, LEGAL-BERT (pre-trained partly on EDGAR contracts, so a natural fit), `max_len` = 256.
4. **Extra metrics:** **clause recall** (share of gold spans flagged with the correct category in at least one overlapping sentence) and **contract-level presence** (does this contract contain clause X?). These don't depend on the sentence-conversion rule.
5. **Inference:** `analyze --type contract agreement.pdf`. Adjacent flagged sentences with the same label are merged into one finding.

| Contract category | Contracts with it | Risk group | Default severity |
|---|---|---|---|
| Uncapped Liability | 111 | Legal risk | High |
| Covenant Not To Sue | 100 | Legal risk | High |
| Non-Compete | 119 | Legal risk | High |
| Exclusivity | 180 | Money & commitments | High |
| Minimum Commitment | 165 | Money & commitments | High |
| Liquidated Damages | 61 | Money & commitments | High |
| Ip Ownership Assignment | 124 | Money & commitments | High |
| Cap On Liability | 275 | Legal risk | Medium |
| Termination For Convenience | 183 | Money & commitments | Medium |
| Renewal Term | 176 | Money & commitments | Medium |
| Notice Period To Terminate Renewal | 111 | Money & commitments | Medium |
| Change Of Control | 121 | Legal risk | Medium |
| Rofr/Rofo/Rofn | 85 | Money & commitments | Medium |
| Volume Restriction | 82 | Money & commitments | Medium |
| No-Solicit Of Employees | 59 | Legal risk | Medium |
| Anti-Assignment | 374 | Legal risk | Low |
| Audit Rights | 214 | Legal risk | Low |
| Insurance | 166 | Money & commitments | Low |
| Governing Law | 437 | Legal risk | Low |

**Done when:** converter golden tests and the span-coverage check pass; TF-IDF is trained and logged with sentence-level metrics, clause recall and contract-level metrics, and the transformers pass `--smoke`; `analyze --type contract` works on a real contract PDF from the CUAD Zenodo zip.

---

### Part 4: Product layer (explanations, API and UI, optimisation, robustness, monitoring, deployment)

1. **Risk score**, separately for each module. For each risky category, take the highest confidence among its findings, *p_c*. Then `raw = Σ w_c · p_c` with severity weights High = 3, Medium = 2, Low = 1. For privacy, each missing protection adds 1. Then `score = 100 · (1 − e^(−raw / k))`, with *k* calibrated per module so a typical validation document scores about 50. Bands: < 34 Low, 34–66 Medium, ≥ 67 High. The score doesn't depend on document length. Check: how well it ranks against the gold-label category count on test documents, plus a quick manual rating of about 10 documents per module.
2. **Explanations:** templates for every label (written in Parts 1–3) are always available. Optional LLM layer (`CLAUSEGUARD_LLM=anthropic`): Claude through the Anthropic API, `claude-opus-5-5` by default with low effort (configurable, e.g. `claude-haiku-5-5` to cut cost), with Anthropic's server-side refusal fallback. It receives only the flagged clauses, their categories and the category definitions in one request per document, and returns schema-constrained JSON. Checks: at most 60 words and no numbers that aren't in the clause; on any error, timeout or refusal, fall back to the template. Results are cached. Human rating of 30 explanations per module for faithfulness and clarity (1–5).
3. **Optional zero-shot LLM comparison** on about 200 test units per module: accuracy, run-to-run consistency, cost, latency.
4. **Optimisation:** ONNX export + dynamic INT8 quantisation, batching, `max_len` tuning, result cache (hash of text + document type + model version). Measure the F1 change and median/p95 latency for short, medium and long documents on your Mac's CPU.
5. **API** (FastAPI): `GET /health`, `GET /document-types`, `POST /analyze` (**`document_type` is required**; a missing or invalid value returns a clear 422 error), `POST /feedback`.
6. **Web UI** (plain HTML/CSS/JavaScript served by the same FastAPI app, instead of Gradio: no heavy dependency, and the highlighted full-document view needs custom rendering): Step 1 radio buttons "What is this document?", Step 2 upload or paste, Step 3 report with highlighted text, findings grouped by risk group, score, protections (privacy), and the "not legal advice" disclaimer.
7. **Failure handling:**

| Case | Response |
|---|---|
| No document type chosen | UI requires a choice; API returns 422 |
| Empty or very short input (< 50 words) | Ask for a longer document |
| Scanned or image-only PDF | Explain that a text-based PDF is needed |
| Non-English text | Explain that only English is supported |
| Very large document | Process in batches up to a documented limit; above it, show a clear message |
| Chosen analyzer recognises almost nothing | Soft note suggesting the user check the type they chose. **No automatic switching.** |
| LLM unavailable or invalid output | Template explanation; detection unaffected |
| Model failed to load | `/health` and `/analyze` return 503 |

8. **Robustness suite** (pytest + report), per module: paraphrases (about 20 hand-written pairs), negation pairs (most important for privacy), formatting noise (all caps, numbering, hyphenation, extra whitespace), typos, unrelated text such as a news article (should produce few flags), very long documents (50k+ words finish without timing out), real PDFs. Metrics: how often predictions flip, and recall on perturbed positives.
9. **Monitoring** (metadata only, never document text): JSON-lines log with request ID, document type, model version, input length, latency per stage, flags per category, share of low-confidence predictions, errors and feedback. `clauseguard monitor summarize` compares these against training-time baselines to detect drift.
10. **Deployment:** Dockerfile → Hugging Face Spaces (CPU). Models are pulled from HF Hub model repos at startup. OPP-115 is licensed for research and teaching only, so keep the privacy model repo private or clearly marked non-commercial.
11. **Final evaluation report** generated from `experiments/log.csv` and the per-module reports.

**Done when:** API and UI tests pass; the robustness, ONNX and latency code runs end to end on the TF-IDF and smoke models; the Docker image builds and runs locally; `make train-gpu` (the script for the GPU session) passes its own smoke mode. Final numbers, ONNX benchmarks against the latency target (detection under 10 seconds for a 5,000-word document on CPU) and the Spaces deployment come after the GPU session (section 6).

---

## 5. GPU requirements

| Where | What runs there |
|---|---|
| **Your Mac (M2, 8 GB)** | Downloads, conversion, data cards, TF-IDF baselines, all `--smoke` runs, inference, ONNX export, API/UI, robustness tests. 8 GB of shared memory isn't enough to fine-tune base-size transformers in reasonable time. |
| **GPU (Lightning AI)** | Fine-tuning DistilBERT, LEGAL-BERT and PrivBERT for all three modules in one session, after all four parts are coded. |

- **Recommended:** 1× **NVIDIA L4 (24 GB)**. An A10G is equally fine. A T4 (16 GB) works at roughly half the speed. An A100 or H100 isn't needed.
- **Rough training time on an L4:** ToS about 5 min per run, privacy about 10 min per run, contracts about 30–45 min per run (with negative sampling). Allow **about 8–10 GPU hours in total** across all experiments, seeds and reruns.
- **When:** once, after Part 4 is finished and every `--smoke` run passes on your Mac.

---

## 6. After Part 4: the training session on Lightning AI

1. Start an L4 Studio, then run `git clone <repo>` and `uv sync --group train`.
2. `make data`: downloads and prepares all three datasets (a few minutes; same code and checksums as on the Mac).
3. `make train-gpu`: trains DistilBERT, LEGAL-BERT and PrivBERT (privacy only) for every module, picks the best per module on validation, re-trains the winner with 3 seeds, tunes thresholds, and writes everything to `artifacts/` and `experiments/log.csv`. The script resumes where it left off if the Studio stops.
4. `make evaluate-final`: runs the test-set evaluation for the chosen models.
5. Copy `artifacts/` and `experiments/log.csv` back (or push them to the HF Hub), then switch the GPU off.
6. Back on the Mac: calibrate the risk score, export to ONNX, run the robustness suite and latency benchmarks, write the final report, and deploy to Spaces.

---

## 7. Later: combining the modules (outside these four parts)

The design already supports this. Every module produces the same `Finding` format and unit tables with character offsets, and labels are namespaced (`tos:`, `privacy:`, `contract:`). Combining can then mean a combined report (run several modules on one document) or a joint model trained on the sentence-level tables (UNFAIR-ToS is sentence-level, the privacy sentence table comes from Part 2, and CUAD sentences come from Part 3). Neither option requires redoing the data work.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rare labels (Arbitration 28 training sentences, Do Not Track 31 segments) | `pos_weight`, per-label thresholds, macro-F1 reported next to micro-F1, results averaged over several seeds |
| Small privacy test set (about 17 policies) | Policy-level stratified split plus 5-fold cross-validation for the baseline; report variance |
| Label noise when converting spans to sentences | Overlap rule, ambiguous sentences left out of training, coverage check, manual sample in each data card |
| Training text looks different from real documents (lowercased ToS text, EDGAR formatting) | Shared normalisation and cleaning used for both training and inference; robustness tests on real PDFs |
| "We do not share your data" flagged as sharing | `negated_practice` sub-label plus negation robustness tests |
| GPU cost or time | `--smoke` locally first; small models; fp16; checkpoints; contract negative sampling |
| OPP-115 licence | Non-commercial use only; privacy model repo kept private or clearly labelled |
