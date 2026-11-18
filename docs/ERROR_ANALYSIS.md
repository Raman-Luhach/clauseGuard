# Error analysis (November 2026)

Validation errors of the three served models after the first GPU session, and what was changed
because of them. Test data was not looked at; every decision below uses the validation split.

| Module | Served run | Validation units | Units with an error |
|---|---|---|---|
| ToS | `tos-legal-bert-s13-20261024-033206` (LEGAL-BERT) | 2,275 sentences | 115 |
| Privacy | `privacy-privbert-s87-20261024-054534` (PrivBERT) | 652 segments | 206 |
| Contract | `contract-legal-bert-s87-20261027-155847` (LEGAL-BERT) | 13,974 sentences | 470 |

The error tables are `errors_validation.csv` in each run folder (one row per unit with a false
positive or false negative, with gold labels, predictions and scores). Counts by error type
below come from simple rules (fragments: text starting with a redaction marker or a lowercase
word and under 12 words, or ending with "["; wrong label: a false positive and a false negative
on the same unit; confident false positive: score >= 0.9); the descriptions come from reading
a random sample of 20-40 errors per module.

## Counts by error type

| Error type | ToS | Privacy | Contract |
|---|---|---|---|
| Fragment from sentence splitting | 5 | 0 | **176** |
| Wrong label (related labels confused) | 7 | **46** | 2 |
| Confident false positive (score >= 0.9) | **48** | 2 | 118 |
| Other false positive | 33 | 74 | 7 |
| Miss (false negative) | 22 | 84 | **167** |
| *(unit contains a negation word)* | 52 | 101 | 131 |

(ToS counts are after the threshold change below, which trades misses for false positives.)

## Terms of Service

- **Most false positives describe the flagged clause type correctly** (about 12 of 14 sampled):
  "Vimeo shall not be responsible for the loss of such content" (limitation of liability), "you
  and Headspace elect to have claims and disputes resolved by arbitration" (arbitration). The
  annotators of UNFAIR-ToS marked clauses they judged *potentially unfair*; similar sentences
  judged fair, or missed, count as negatives. The model learns the clause type more than the
  fairness judgement, which for a reader who wants to know "where are the arbitration
  clauses?" is acceptable.
- **Misses are implicit wording** ("if your account is terminated … you will not be refunded",
  "you agree not to hold us responsible for banking charges"), **list items without their lead-in**
  ("limit your access to or use of the Airbnb platform;"), and **related labels**
  (termination vs. content removal: "termination … can include disabling your access to … any
  content you submitted").
- Validation recall (0.865) overstated test recall (0.766 over three seeds): the UNFAIR-ToS
  splits are chronological, and the later test documents phrase things differently.

**Change:** recall-targeted thresholds. Instead of F1-optimal thresholds, the smallest F-beta
whose validation any-risk recall reaches 0.90 is used, as long as any-risk precision stays at or
above 0.70 (`target_any_recall`, `min_any_precision` in `configs/tos.yaml`; `clauseguard retune`).
That is F2 for the served model. No retraining was needed:

| Served ToS model | Validation recall / precision / macro-F1 | Test recall / precision / macro-F1 |
|---|---|---|
| F1 thresholds (before) | 0.865 / 0.816 / 0.820 | 0.826 / 0.816 / 0.762 |
| F2 thresholds (after) | 0.913 / 0.747 / 0.765 | **0.872 / 0.781 / 0.770** |

The proposal's targets (recall >= 0.85, precision >= 0.70) are now met on test.

## Privacy policies

- **The catch-all category "Other"** is the largest source of errors (26 misses, 18 false
  positives, 9 confused with First Party Collection). It covers introductions, contact details
  and generic statements, and annotators used it inconsistently.
- **Category boundaries** that the annotators themselves disagree on: public forums ("anything you
  post becomes public") predicted as Third Party Sharing; ad-network paragraphs labelled Other
  but predicted as sharing; collection vs. sharing in paragraphs that mention both.
- **Negation** is mostly handled: the dedicated sub-label (`privacy:negated_practice`) lowers the
  severity of statements like "we do not sell your personal information"; the negation pairs in
  the robustness suite pass.
- Any-risk recall is already high (0.914 on test), so no threshold change.

**Change:** none to the data or thresholds; the second GPU session adds oversampling of rare
labels (Do Not Track has 22 training segments) and DeBERTa-v3 challengers for macro-F1.

## Contracts

- **A third of all errors (176 of 470) were sentence fragments** created by redaction markers.
  CUAD contracts were redacted before filing ("at least [***] units"), and the sentence splitter
  broke sentences at the brackets: "***] to produce such a product.", "*] ([*] dollars) per
  month.". Labels were then attached to meaningless fragments in training, and real clauses
  were cut in half at prediction time. 127 of the 510 contracts are affected.
- **Confident false positives are often real clauses that weren't annotated** (about 8 of 14
  sampled): a governing-law clause inside an attached assignment ("This Assignment shall be
  governed by the laws of the State of Delaware"), insurance and exclusivity statements. CUAD's
  annotation is not exhaustive for every repetition.
- **Misses**: fragments (above), label inheritance from long spans (a list item inside an
  annotated exclusivity span: "(vii) recording sales of the Product in the Territory in its
  books"), definitions ("'Change of Control' shall mean…"), and paraphrases ("Any Participant
  will have the right to voluntarily withdraw … at any time" for termination for convenience).
- High thresholds (often above 0.95) caused by these noisy negatives push recall down: some
  misses score 0.91-0.97.

**Change 1: sentence splitting** (`src/clauseguard/ingestion/segment.py`): never split inside an
open bracket or before the rest of a redaction marker. The contract unit table was rebuilt:

| Contract sentences | Before | After |
|---|---|---|
| Sentences | 117,613 | 111,759 |
| Fragments around redaction markers | 8,095 | 617 |
| Labelled fragments | 707 | 16 |
| TF-IDF baseline, validation macro-F1 / any-risk recall | 0.578 / 0.578 | **0.660 / 0.692** |

(The validation sentences changed as well, so before/after is not a strict comparison; it is the
same model and settings on cleaner units.)

The served LEGAL-BERT, without retraining, scored on the rebuilt sentences and then given
recall-targeted thresholds (F-1.25: the largest beta keeping precision >= 0.70):

| Served contract model | Validation macro-F1 / recall / precision | Test macro-F1 / recall / precision |
|---|---|---|
| Old sentences, F1 thresholds | 0.638 / 0.686 / 0.694 | 0.606 / 0.611 / 0.726 |
| Rebuilt sentences, F1 thresholds | 0.689 / 0.739 / 0.755 | – |
| Rebuilt sentences, F-1.25 thresholds | 0.708 / 0.766 / 0.735 | **0.625 / 0.664 / 0.703** |

Better, but still far from the 0.85 recall target: the model itself was trained on fragmented
sentences, so contracts are retrained in GPU session 2.

**Change 2: GPU session 2** trains on the rebuilt sentences with twice as many unlabelled
sentences, with and without oversampling of rare clause types, for LEGAL-BERT and DeBERTa-v3
(a 2x2 comparison), with recall-targeted thresholds (target 0.85, precision >= 0.70). On the
current model, thresholds alone could not reach the target without precision falling to 0.43,
so the improvement has to come from training.

## What oversampling did for the TF-IDF baselines

A free check before using GPU time: repeating rare labels' units (up to 4x) did not help the
TF-IDF models (ToS validation macro-F1 0.747 → 0.730, contracts 0.660 → 0.657), which already
use class weights. The transformers use a milder weighting (square-root), so the GPU session
still tests oversampling, with and without it, rather than assuming either way.

## Latency (found while checking the 10-second target)

On the Mac a 5,000-word document took 17-19 s for ToS and contracts (the earlier 10-11 s were
measured on Kaggle's CPU). Two changes, neither of which changes any prediction:

- **Screening** (`clauseguard screening`, `src/clauseguard/models/screening.py`): the TF-IDF
  model scores every unit in milliseconds, and the transformer reads only units it gives some
  chance of a label. The cut-off keeps >= 99% of the transformer's own flags on validation:
  ToS reads 45% of sentences, contracts 35%; validation recall unchanged (0.913 and 0.766).
  Privacy is not screened: nearly every paragraph carries some label.
- **Batching by token length** (`OnnxModel.predict_proba`): texts are sorted by token count and
  run in batches of 4, each cut to its own longest text. On a CPU the work is proportional to the
  padded tokens; batches of 32 sorted by characters wasted most of it on padding.

See `reports/generated/<module>_benchmark.md` for the measured times. ONNX Runtime's transformer
graph optimizer was also tried: no gain (the export pattern prevents attention fusion), so it is
only kept when it measures faster.
