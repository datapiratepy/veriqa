# VeriQA
### Selective Question Answering over Document Collections using Retrieval-Grounded Confidence Estimation

VTU Major Project · Dept. of Computer Science & Engineering · AMC Engineering College · 2026

**Harsh Kamat** · **Harsh Ranjan** · **Manish Kumar**

---

## 1. What it does

VeriQA answers questions from a document collection and **declines to answer when its own
answer is likely to be wrong.** It retrieves passages, extracts a candidate span, computes
sixteen reliability signals from the retrieval and reading stages, converts them to a calibrated
probability of error, and abstains when that probability exceeds a threshold chosen to hit an
accuracy target.

Everything runs on a laptop CPU. No GPU, no API key, no network call at query time, and no
pretrained neural weights.

## 2. Research question and answer

> Do retrieval-side evidence signals improve prediction of answer error beyond reader-side
> signals alone in a CPU-only extractive question-answering pipeline?

**Yes, measurably.** On 3,585 held-out SQuAD 2.0 questions across 36 unseen articles:

| | Error-prediction ROC-AUC | AURC |
|---|---|---|
| Reader-only calibrator (B5, after Kamath et al.) | 0.5242 | 0.8785 |
| **Fused retrieval + reader + agreement (P1)** | **0.6390** [0.6127, 0.6668] | **0.8423** |

ΔAURC = **−0.0345** [−0.0495, −0.0191], paired bootstrap over articles, interval excludes zero.
Holds at correctness thresholds 0.4, 0.5 and 0.6. Reliability layer costs **1.96 ms**.

Ablation attributes the gain to retrieval (ΔAURC 0.0274) and agreement (0.0103); removing all
five **reader** features changes nothing significant (0.0067, CI includes zero).

## 3. Honest status

| Gate | Criterion | Measured | |
|---|---|---|---|
| G0 | negative control ∈ [0.45, 0.55] | 0.4887 | PASS |
| G1 | base reader accuracy ≥ 0.45 | 0.1144 | **FAIL** |
| G2 | some method AUC ≥ 0.70 | 0.6390 | **FAIL** |
| G3 | P1 AURC < B5, CI excludes 0 | −0.0345 | PASS |
| G4 | P1 AURC < B4, CI excludes 0 | −0.0293 | PASS |
| G5 | ECE < 0.10 and target within ±3 pts | 0.0120 / gap −0.52 | **FAIL** |
| G6 | reliability layer < 100 ms | 1.96 ms | PASS |

**Four pass, three fail.** The comparative hypothesis passes. The absolute system is weak
because the reader is a classical span scorer, not a transformer — Hugging Face was unreachable
in the build environment. See `docs/implementation/READER_BACKEND.md`. Absolute numbers are
low and are reported as such everywhere.

## 4. Layout

```
VeriQA/
├── configs/default.yaml        seed 20260814, all hyperparameters
├── data/raw/                   SQuAD 2.0 + SHA-256 checksums (read only)
├── data/processed/             built corpora A and B
├── src/veriqa/
│   ├── ingestion/  retrieval/  reader/  reliability/  baselines/  evaluation/
│   ├── service.py              shared query service
│   └── api/main.py             FastAPI backend
├── app/streamlit_app.py        demo UI
├── experiments/01..09          corpus → pipeline → E0–E8 → figures + freeze
├── results/tables/             T1–T10, every paper number
├── results/figures/            F2–F6
├── results/RESULT_FREEZE.json  frozen, with gates and file hashes
├── tests/                      19 tests incl. leakage and hand-computed metrics
├── docs/                       protocol, architecture, reproduction, demo, viva, audit
└── paper/                      IEEE manuscript (.md and .tex)
```

## 5. Quick start

```bash
pip install -r requirements.txt
python3 experiments/01_build_corpus.py       # corpora + index + reader   (~1 min)
python3 experiments/02_run_pipeline.py A     # resumable, ~8 min
python3 experiments/02_run_pipeline.py B     # ~1 min
python3 experiments/04_train_evaluate.py     # E0 negative control, then E1–E4
python3 experiments/05_ablation.py           # E5
python3 experiments/06_domain_shift.py       # E6
python3 experiments/07_latency.py            # E7
python3 experiments/08_error_analysis.py     # E8 + sensitivity
python3 experiments/09_figures_and_freeze.py # figures + RESULT_FREEZE.json
python3 -m pytest tests -q                   # 19 tests
streamlit run app/streamlit_app.py           # demo
uvicorn veriqa.api.main:app --app-dir src    # API
```

## 6. Integrity notes

1. Every number in the paper traces to `results/tables/*.csv` or `results/RESULT_FREEZE.json`.
   A programmatic audit verifies 65 of them; see `docs/FINAL_PROJECT_AUDIT.md`.
2. The trained-calibrator idea is **Kamath, Jia & Liang (ACL 2020)**, reimplemented here as
   baseline B5 and credited as theirs. TF-IDF, SVD, BM25, gradient boosting and isotonic
   regression are standard methods used as published.
3. **No claim of IEEE formatting compliance.** `IEEEtran.cls` was not available in the build
   environment, so the `.tex` has not been compiled against the official template. The body
   compiles cleanly to 7 pages with a stand-in two-column class
   (`paper/VERIQA_MANUSCRIPT_LAYOUT_PREVIEW.pdf`) — that is a length sanity check, not a
   compliance check.
4. **References [2]–[10] are unverified.** They were located by literature search and must be
   opened and checked by a human before submission. [1] was confirmed against the ACL Anthology.
5. Nothing here is published or peer reviewed.
