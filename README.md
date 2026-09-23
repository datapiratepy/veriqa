# VeriQA

### Selective extractive question answering: knowing when not to answer

VTU major project (B.E. Computer Science and Engineering), AMC Engineering College,
Bengaluru, 2026.

**Team:** Harsh Kamat · Harsh Ranjan · Manish Kumar
**Project guide:** Divya Tyagi · **Project coordinator:** Snigdha Kesh

---

## 1. What it is

VeriQA answers questions from a document collection and **declines to answer when its own
answer is likely to be wrong**. It retrieves passages, extracts a candidate answer span,
computes sixteen reliability signals from the retrieval and reading stages, turns them into
a calibrated probability of error, and abstains when that probability exceeds a threshold
fitted on held-out calibration data. When it abstains, it says which signals triggered the
refusal.

It runs entirely on a laptop CPU: no GPU, no API key, no network call at query time, and no
pretrained neural weights. It comes with a Streamlit demo and a FastAPI backend.

## 2. The research problem

A question-answering system that never declines will eventually present a wrong answer
with the same confidence as a right one. Training a separate model to predict the
reader's errors is an established idea (Kamath, Jia & Liang, ACL 2020). What had not been
isolated is how much the **retrieval stage** adds to that prediction once a reader-side
calibrator is already in place:

> Do retrieval-side evidence signals improve prediction of answer error beyond reader-side
> signals alone, in a CPU-only extractive question-answering pipeline?

The contribution is the measurement, not a new mechanism. The Kamath et al. calibrator is
reimplemented here as baseline B5 and credited as theirs.

The evaluation uses SQuAD 2.0. Because SQuAD's "unanswerable" labels are
paragraph-relative, adding retrieval can silently invalidate them. The team measured this
(14.89% of substantive unanswerable questions had their plausible answer elsewhere in an
article-pooled corpus) and adopted **sibling masking**, which cuts it to 2.39%
([protocol](docs/research/SQUAD_ANSWERABILITY_PROTOCOL.md)).

## 3. Architecture

```
question
   │
Hybrid retrieval        TF-IDF → TruncatedSVD (256-d LSA) + BM25Okapi,
   │                    min-max fused (α = 0.5), sibling-masked, exact top-k (k = 5)
   │
Extractive reader       1. keep the 3 sentences with the highest question overlap
   │                    2. enumerate spans ≤ 8 tokens, drop stopword/punctuation edges
   │                    3. gradient-boosted span scorer over 20 span features
   │                    4. gradient-boosted null head → P(no answer in this passage)
   │
Reliability layer       16 features: retrieval (7) + reader (5) + agreement (4)
   │                    HistGradientBoosting → isotonic calibration → P(error)
   │                    threshold chosen on the calibration split only
   │
risk > threshold ? ── yes ──► abstain, with the triggering reasons
   │
   no ──► answer + supporting passage + highlighted span + risk
```

Evaluation discipline: splits are made **by article** (40/20/10/30 across reader
training, reliability training, calibration and test) in a single function, and
`tests/test_no_leakage.py` asserts the four partitions are disjoint. Confidence intervals
come from a bootstrap that resamples **articles**, not questions. A **negative control**
(shuffled error labels) must land at chance before any other result is reported, and the
results are judged against seven pass/fail gates, reported below whether or not they passed.

Module-by-module detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Design decisions,
including two bugs caught and fixed before the results were frozen:
[docs/DECISIONS_LOG.md](docs/DECISIONS_LOG.md).

## 4. My contribution (Harsh Kamat)

My part of the system is the **reader** and the **reliability layer**, the component that
decides whether to answer, plus the baselines it is measured against:

| Module | What it does |
|---|---|
| [`reader/spans.py`](src/veriqa/reader/spans.py) | Candidate-span generation and the 20 vectorised span features |
| [`reader/extractive.py`](src/veriqa/reader/extractive.py) | Two-stage reader: sentence selection, gradient-boosted span scorer, trained null-answer head, and the `read()` interface |
| [`reliability/features.py`](src/veriqa/reliability/features.py) | The 16 reliability features across the retrieval, reader and agreement families |
| [`reliability/calibrator.py`](src/veriqa/reliability/calibrator.py) | Gradient-boosted risk model, isotonic calibration, threshold selection on calibration data |
| [`baselines/methods.py`](src/veriqa/baselines/methods.py) | Baselines B1–B5 (B5 reimplements Kamath et al.) and the fitting of the proposed calibrator P1 |

**Harsh Ranjan** owned ingestion and corpus construction (`ingestion/squad.py`) and the
hybrid retriever with sibling masking (`retrieval/hybrid.py`). **Manish Kumar** owned the
evaluation layer (`evaluation/splits.py`, `metrics.py`, `squad_eval.py`), the query service,
the FastAPI backend and the Streamlit app.

This split follows the team's module map in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The repository was published as a single squashed commit, so per-file authorship is
recorded there rather than in the git history.

## 5. Results

Frozen on 2026-08-13 (seed `20260814`) in
[`results/RESULT_FREEZE.json`](results/RESULT_FREEZE.json). Test set: **3,585 questions
from 36 articles** that neither model saw during training. Brackets are 95% bootstrap
intervals over articles.

| Method | Error-prediction ROC-AUC | AURC (lower is better) |
|---|---|---|
| B4: reader's own null-answer score | 0.5526 | 0.8683 |
| B5: reader-only calibrator (after Kamath et al.) | 0.5242 | 0.8785 |
| **P1: retrieval + reader + agreement** | **0.6390** [0.6127, 0.6668] | **0.8423** [0.8029, 0.8731] |

- P1 reduces AURC relative to B5 by **0.0345** (ΔAURC −0.0345 [−0.0495, −0.0191], paired
  bootstrap; the interval excludes zero), and relative to B4 by 0.0293 [0.0150, 0.0442].
  The result holds at correctness thresholds F1 ≥ 0.4, 0.5 and 0.6.
- **Ablation** attributes the gain to retrieval features (ΔAURC 0.0274 [0.0149, 0.0411])
  and agreement features (0.0103 [0.0043, 0.0173]). Removing all five reader features
  changes nothing measurable (0.0067 [−0.0044, 0.0189]; the interval includes zero).
- **Calibration:** isotonic regression lowers expected calibration error from 0.0895 to
  0.0120.
- **Cost:** the reliability layer adds a median **1.96 ms** per query (host-dependent);
  the median end-to-end query takes 63.8 ms on CPU.
- **Negative control:** 0.4887 with shuffled labels, inside the required [0.45, 0.55] band.

### Pass/fail gates: 4 pass, 3 fail

| Gate | Criterion | Measured | |
|---|---|---|---|
| G0 | Negative control within [0.45, 0.55] | 0.4887 | PASS |
| G1 | Base reader accuracy ≥ 0.45 | 0.1144 | **FAIL** |
| G2 | Some method reaches error-prediction AUC ≥ 0.70 | 0.6390 | **FAIL** |
| G3 | P1 AURC < B5, interval excludes 0 | −0.0345 | PASS |
| G4 | P1 AURC < B4, interval excludes 0 | −0.0293 | PASS |
| G5 | ECE < 0.10 **and** accuracy target met within ±3 points | 0.0120, gap −0.52 | **FAIL** |
| G6 | Reliability layer < 100 ms | 1.96 ms | PASS |

The comparative hypothesis (G3, G4) passes. The absolute system is weak, and the failed
gates say so: base reader accuracy is 11.4%, and no method reaches 80% accuracy on more
than 0.2% of test questions.

Tables T1–T10 are in [`results/tables/`](results/tables/) and figures F2–F6 in
[`results/figures/`](results/figures/).

## 6. Limitations

- **The reader is classical, not a transformer, and it bounds everything.** The design
  called for a DistilBERT reader and MiniLM embeddings. Pretrained weights could not be
  downloaded in the environment where the experiments were built and run, so every
  reported number comes from a feature-based reader and LSA retrieval
  ([decision D-009](docs/implementation/READER_BACKEND.md)). Mean answer F1 on answerable
  questions is 0.145. The retrieval-over-reader finding was measured with a weak reader
  and may narrow with a strong one.
- **The retriever is not competitive with neural retrieval** (89.1% top-5 gold recall on
  the test set), and there is no neural-retriever baseline.
- **Dataset scope:** English Wikipedia paragraphs from SQuAD 2.0, whose unanswerable
  questions were written adversarially and may not resemble naturally occurring ones;
  2.39% residual label risk remains after sibling masking.
- **Scale and transfer:** 5,637 passages with an exact index; behaviour at millions of
  passages is untested. The single transfer test (to a disjoint corpus built from SQuAD
  2.0 dev) changes the corpus and the reader instance together, so it is weak evidence.
- **The risk score is coarse.** Isotonic calibration is piecewise-constant, so the
  abstention gate behaves almost like a binary switch in the demo.
- **The demo answers over the fixed SQuAD corpus.** There is no document-upload feature.
- No deployment or user study.

## 7. Run it

Python 3.10 (tested on 3.10.12). No GPU and no model downloads.

```bash
pip install -r requirements.txt
cd data/raw && sha256sum -c CHECKSUMS.txt && cd ../..   # verify the SQuAD files
python -m pytest tests -q                               # 19 tests
streamlit run app/streamlit_app.py                      # demo UI
uvicorn veriqa.api.main:app --app-dir src               # API: /health, /ask, /evaluate
```

The trained artefacts in `models/` and the corpora in `data/processed/` ship with the
repository, so the demo runs without retraining. Example questions that answer and that
abstain: [docs/demo/DEMO_SCREENSHOT_GUIDE.md](docs/demo/DEMO_SCREENSHOT_GUIDE.md).

**Reproduce the experiments** (about 15–20 minutes end to end on CPU, most of it in
`02_run_pipeline.py A`):

```bash
python experiments/01_build_corpus.py        # corpora, retrieval index, reader
python experiments/02_run_pipeline.py A      # resumable; delete results/raw/predictions_*.jsonl for a clean run
python experiments/02_run_pipeline.py B
python experiments/04_train_evaluate.py      # E0 negative control first, then E1–E4
python experiments/05_ablation.py            # E5
python experiments/06_domain_shift.py        # E6
python experiments/07_latency.py             # E7
python experiments/08_error_analysis.py      # E8 and threshold sensitivity
python experiments/09_figures_and_freeze.py  # figures + a new RESULT_FREEZE.json
```

There is no script `03`; feature extraction happens inside `02`. Every statistical result
is deterministic under the fixed seed; only latency varies with hardware.

**Reproduction check (September 2026):** a clean re-run of the whole pipeline from the raw
SQuAD files, on Python 3.10.20, produced the same predicted answer for all 7,185 corpus-A
questions and every frozen result table (T1–T6, T8–T10) byte for byte. Only the latency
table (T7) differed.

Details: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) · [docs/SETUP.md](docs/SETUP.md)

## 8. Project status

- **Implementation and experiments complete**, with results frozen on 2026-08-13.
- **Paper:** an IEEE-format manuscript, *Quantifying the Incremental Value of
  Retrieval-Side Reliability Signals for Selective Extractive Question Answering*, is in
  [`paper/`](paper/) ([PDF](paper/VERIQA_IEEE_FINAL.pdf)). It is **not published and has
  not been peer reviewed**.
- **The most valuable next step** is re-running the measurement with a transformer reader.
  The reader sits behind one interface (`read()` returns a `ReadOut`), so a transformer can
  be dropped in without touching the reliability layer.

## Repository layout

```
configs/default.yaml     seed and every hyperparameter (single source of truth)
data/raw/                SQuAD 2.0 train/dev + SHA-256 checksums
data/processed/          built corpora A (main) and B (transfer)
src/veriqa/              ingestion · retrieval · reader · reliability · baselines · evaluation
                         service.py (shared query service) · api/ (FastAPI)
app/streamlit_app.py     demo UI
experiments/             01, 02, 04–09: corpus → pipeline → E0–E8 → figures + freeze (no 03)
models/                  trained artefacts used by the demo and the experiments
results/                 RESULT_FREEZE.json, tables T1–T10, figures, raw outputs
tests/                   19 tests: metrics against hand-computed values, features, retrieval, leakage
paper/                   IEEE-format manuscript (.tex, .pdf, .bib) and its validation records
docs/                    architecture, decisions, protocol, reproducibility, demo, viva
```

## Data and credits

- **SQuAD 2.0** (Rajpurkar, Jia & Liang, 2018) is redistributed in `data/raw/` under its
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, from the
  [SQuAD project](https://rajpurkar.github.io/SQuAD-explorer/).
- The trained-calibrator approach behind baseline B5 is from Kamath, Jia & Liang,
  *Selective Question Answering under Domain Shift* (ACL 2020). TF-IDF, truncated SVD,
  BM25, gradient boosting and isotonic regression are standard methods used as published.
