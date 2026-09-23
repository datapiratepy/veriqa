# VeriQA — Implementation Handoff

**Date:** 17 August 2026 · **Audience:** Harsh Kamat, Harsh Ranjan, Manish Kumar
**Purpose:** an honest inventory of what actually exists, and exact instructions to run it on Windows.

> **Note (September 2026):** moved here from the repository root. File paths below describe the
> layout when this handoff was written; `DEMO_SCREENSHOT_GUIDE.md` now lives in `docs/demo/`,
> and `docs/RESEARCH_PROTOCOL.md` was a duplicate of `docs/research/SQUAD_ANSWERABILITY_PROTOCOL.md`.
> Discrepancies 1–3, known issue 4 and item K(ii) below describe earlier drafts and are resolved:
> `docs/ARCHITECTURE.md` lists no experiment scripts, `docs/demo/DEMO_GUIDE.md` describes no
> upload or side-by-side pane, and Fig. 1 of the final paper shows no PDF/TXT/MD ingestion.
> Document upload and PDF/TXT/MD ingestion themselves still do not exist.

> **Evidence levels used throughout this document.** Every claim is tagged.
> **[VERIFIED]** — I executed it in this session and observed the output.
> **[INFERRED]** — read from source code, not executed.
> **[DOC]** — asserted in the paper or docs only.

---

## PART 1 — What was actually built

### A. What the system is

A CPU-only selective extractive question-answering system. It retrieves passages from a
fixed corpus, extracts a candidate answer span, computes 16 reliability features, converts
them to a calibrated probability that the answer is wrong, and either returns the answer or
abstains.

It is **not** a chatbot, not generative, and does not use any transformer or pretrained
neural weights.

### B–D. Components that exist, their files, and whether they are executable code

| Component | File | Real code? | Used for IEEE results? |
|---|---|---|---|
| Config loader (+ mini-YAML parser) | `src/veriqa/config.py` (86 ln) | ✅ | ✅ |
| Seeding | `src/veriqa/utils/seeds.py` (9 ln) | ✅ | ✅ |
| SQuAD loading, corpus construction | `src/veriqa/ingestion/squad.py` (116 ln) | ✅ | ✅ |
| Hybrid retrieval (TF-IDF→SVD + BM25, sibling mask) | `src/veriqa/retrieval/hybrid.py` (81 ln) | ✅ | ✅ |
| Candidate spans + 20 span features | `src/veriqa/reader/spans.py` (181 ln) | ✅ | ✅ |
| Two-stage reader + null head | `src/veriqa/reader/extractive.py` (166 ln) | ✅ | ✅ |
| The 16 reliability features | `src/veriqa/reliability/features.py` (99 ln) | ✅ | ✅ |
| Risk model + isotonic + threshold | `src/veriqa/reliability/calibrator.py` (49 ln) | ✅ | ✅ |
| Baselines B1–B5 | `src/veriqa/baselines/methods.py` (45 ln) | ✅ | ✅ |
| Article-level splitter | `src/veriqa/evaluation/splits.py` (55 ln) | ✅ | ✅ |
| Metrics (AUC, AURC, ECE, bootstrap) | `src/veriqa/evaluation/metrics.py` (109 ln) | ✅ | ✅ |
| SQuAD EM/F1/normalisation | `src/veriqa/evaluation/squad_eval.py` (47 ln) | ✅ | ✅ |
| End-to-end pipeline runner | `src/veriqa/pipeline.py` (107 ln) | ✅ | ✅ |
| Query service (used by both UIs) | `src/veriqa/service.py` (75 ln) | ✅ | ❌ demo only |
| FastAPI backend | `src/veriqa/api/main.py` (39 ln) | ✅ | ❌ demo only |
| Streamlit UI | `app/streamlit_app.py` (~150 ln) | ✅ | ❌ demo only |

**Total: 2,211 lines of Python across source, experiments and tests.**

### E. Documentation only (no executable content)

`README.md`, `docs/ARCHITECTURE.md`, `docs/SETUP.md`, `docs/REPRODUCIBILITY.md`,
`docs/DECISIONS_LOG.md`, `docs/FINAL_PROJECT_AUDIT.md`, `docs/RESEARCH_PROTOCOL.md`,
`docs/research/SQUAD_ANSWERABILITY_PROTOCOL.md`, `docs/implementation/READER_BACKEND.md`,
`docs/demo/DEMO_GUIDE.md`, `docs/viva/VIVA_GUIDE.md`, and everything in `paper/`.

### F. Frozen experimental results

`results/RESULT_FREEZE.json` (the authoritative freeze, seed 20260814, frozen
2026-08-13T09:28:40), `results/tables/T1..T10*.csv`, `results/raw/E0_*.json`,
`E2_*.json`, `E3_*.json`, `E7_*.json`, `E8_examples.json`, `corpus_build.json`.

### G. Generated outputs (reproducible, not hand-made)

`data/processed/corpus_{A,B}.json`, `models/artifacts_{A,B}.pkl` (57 MB / 21 MB),
`models/reliability_A.pkl`, `results/raw/predictions_{A,B}.jsonl` (5.9 MB / 1.7 MB),
`results/figures/*.png`, `paper/figures/*.png`, `paper/VERIQA_IEEE_FINAL.pdf`.

### H. Simulated, approximated, or manually prepared

**Nothing is simulated and no result was hand-entered.** Two honest approximations:

1. **The reader is not a transformer.** The design called for DistilBERT + MiniLM. Neither
   could be downloaded (Hugging Face blocked in the build environment), so the reader is a
   classical feature-based span scorer and retrieval uses TF-IDF→SVD (LSA). This is
   documented in `docs/implementation/READER_BACKEND.md` and disclosed in the paper.
   **This is the single most important thing to understand about the project.**
2. **Corpus A is a 120-article sample** of SQuAD 2.0 train, not the full 442 articles —
   a runtime decision, recorded in `configs/default.yaml`.

### I–L. External requirements

| Requirement | Needed? |
|---|---|
| External dependencies | Yes — 7 packages for experiments, 4 more for the UI (all PyPI) |
| Internet at **setup** | Yes, once, for `pip install` |
| Internet at **run time** | **No.** [VERIFIED] no network call in any code path |
| Pretrained models | **No.** Nothing is downloaded; everything is fitted from SQuAD itself |
| API keys | **No.** None anywhere in the codebase |
| GPU | **No** |

### M. Fully offline after setup — ✅ [VERIFIED]

### N. What actually produced the IEEE results

`experiments/01_build_corpus.py` → `02_run_pipeline.py` → `04_train_evaluate.py` →
`05_ablation.py` → `06_domain_shift.py` → `07_latency.py` → `08_error_analysis.py` →
`09_figures_and_freeze.py`, all driving `src/veriqa/*`.

**The Streamlit app and FastAPI backend produced none of the reported numbers.** They are
demonstration surfaces over the same trained artefacts.

### ⚠️ Discrepancies between docs and reality — flagged as required

| # | Issue | Reality |
|---|---|---|
| 1 | *(Resolved in the published docs.)* **`experiments/03_extract_features.py` does not exist.** `docs/ARCHITECTURE.md` and the roadmap imply a 9-script chain 01–09. | Feature extraction happens inside `02_run_pipeline.py` via `pipeline.run_questions()`, which calls `features.extract_all()` per question. The numbering skips 03. **No results are affected**; only the documentation is misleading. |
| 2 | *(Resolved in the final paper.)* The paper's Fig. 1 shows "PDF/TXT/MD ingestion". | `src/veriqa/ingestion/squad.py` **only parses SQuAD JSON.** There is no PDF or Markdown loader. The Streamlit app has **no document-upload control.** The paper does not claim upload, but earlier planning docs did. **Do not demo document upload — it does not exist.** |
| 3 | *(Resolved: the current guide has neither.)* `docs/DEMO_GUIDE.md` describes uploading a PDF live and a "side-by-side baseline pane". | Neither exists. Use `DEMO_SCREENSHOT_GUIDE.md` instead — it describes only what is real. |
| 4 | `requirements.txt` pins `pytest==8.*` | Works, but `pip` may warn. Harmless. |

---

## PART 2 — Actual architecture (files, inputs, outputs)

```
SQuAD 2.0 JSON  (data/raw/train-v2.0.json, dev-v2.0.json)
      |  ingestion/squad.py :: load_squad(), build_corpus()
      v      in: JSON  out: Corpus(passages[], questions[])
Corpus + article split      (evaluation/splits.py :: split_articles())
      |
      v  retrieval/hybrid.py :: HybridRetriever.fit()
Index: TfidfVectorizer -> TruncatedSVD(256) -> normalized matrix; BM25Okapi
      |      deps: scikit-learn, rank_bm25, numpy
      v
QUESTION (str)
      |  HybridRetriever.search(question, article, gold_pid, k)
      v      out: Retrieved(pids, dense, bm25, hybrid, all_hybrid)
Top-k passages
      |  reader/extractive.py :: ExtractiveReader.read()
      |    stage 1 spans.py :: candidate_mask()  (top-3 sentences)
      |    stage 2 spans.py :: span_features()   (20 features, vectorised)
      |    scoring HistGradientBoostingClassifier + null head
      v      out: ReadOut(answer, span_prob, null_score, span_len, span_entropy,
      |                   passage_pid, per_passage)
      v  reliability/features.py :: extract_all()
16 features = retrieval_features(7) + reader_features(5) + agreement_features(4)
      |  reliability/calibrator.py :: RiskModel.risk()
      v      GBM -> IsotonicRegression -> P(error)
      |  reliability/calibrator.py :: choose_threshold()  [fitted offline]
      v
risk > threshold ?  ── yes ──> ABSTAIN + reason   (service.py builds the reason string)
                    └─ no ───> ANSWER + passage + evidence + risk
```

---

## PART 3 — File tree by category

```
VeriQA/
├── SOURCE CODE ─────────────────────────────────────────────────
│   src/veriqa/config.py            config loader          (imported)
│   src/veriqa/pipeline.py          orchestration          (imported)
│   src/veriqa/service.py           query service          (imported by both UIs)
│   src/veriqa/ingestion/squad.py   SQuAD JSON only
│   src/veriqa/retrieval/hybrid.py  LSA + BM25 + masking
│   src/veriqa/reader/spans.py      candidates + 20 features
│   src/veriqa/reader/extractive.py span scorer + null head
│   src/veriqa/reliability/features.py    the 16 features
│   src/veriqa/reliability/calibrator.py  GBM + isotonic + threshold
│   src/veriqa/baselines/methods.py       B1–B5
│   src/veriqa/evaluation/{splits,metrics,squad_eval}.py
│   src/veriqa/api/main.py          FastAPI: /health /ask /evaluate
│   app/streamlit_app.py            the UI            [executable entry point]
├── EXPERIMENT SCRIPTS (all executable; produce the paper's numbers) ──
│   experiments/01_build_corpus.py     [03 intentionally absent — see above]
│   experiments/02_run_pipeline.py     (resumable; takes optional seconds budget)
│   experiments/04_train_evaluate.py   E0 negative control, E1, E2, E3, E4
│   experiments/05_ablation.py         E5 + permutation importance
│   experiments/06_domain_shift.py     E6
│   experiments/07_latency.py          E7
│   experiments/08_error_analysis.py   E8 + F1-threshold sensitivity
│   experiments/09_figures_and_freeze.py  figures + RESULT_FREEZE.json
├── DATA ────────────────────────────────────────────────────────
│   data/raw/{train,dev}-v2.0.json  46 MB, read-only, SHA-256 in CHECKSUMS.txt
│   data/processed/corpus_{A,B}.json  generated by 01
│   data/veriqa.sqlite              query log (see Known Issue 1)
├── MODELS (generated) ──────────────────────────────────────────
│   models/artifacts_A.pkl  57 MB   retriever + reader + splits (Corpus A)
│   models/artifacts_B.pkl  21 MB   same for Corpus B
│   models/reliability_A.pkl 1.4 MB risk models + threshold
├── RESULTS ─────────────────────────────────────────────────────
│   results/RESULT_FREEZE.json      ⭐ authoritative
│   results/tables/T1..T10*.csv     every paper number
│   results/raw/*.json, predictions_{A,B}.jsonl
│   results/figures/F2..F6*.png
├── PAPER ── paper/VERIQA_IEEE_FINAL.{tex,pdf,bib} + reports
├── DOCS  ── docs/**  (documentation only)
└── TESTS ── tests/test_{metrics,retrieval,features,no_leakage}.py  (19 tests)
```

---

## PART 4 & 5 — How to run it on Windows (PowerShell)

### ▶ START HERE: **OPTION A — the demo** (~5 minutes)

Everything is pre-built. You do **not** need to retrain anything.

**STEP 1 — open PowerShell in the project folder**
```powershell
cd "C:\Users\harsh\Claude\Projects\IEEE Paper\VeriQA"
```

**STEP 2 — create and activate a virtual environment**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*If activation is blocked:* `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then retry.
*Expect:* your prompt to gain a `(.venv)` prefix.

**STEP 3 — install dependencies** (the only step needing internet)
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*Expect:* ~2–4 minutes, ending in `Successfully installed ...`.

**STEP 4 — data** — nothing to do. `data\raw\*.json` already ship with the repo.
Optional integrity check:
```powershell
Get-FileHash data\raw\train-v2.0.json -Algorithm SHA256
```
Compare against `data\raw\CHECKSUMS.txt`.

**STEP 5 & 6 — corpus and models** — nothing to do. `models\*.pkl` and
`data\processed\*.json` are already built and are what the paper used.

**STEP 7 — smoke test (30 s).** Do this before the UI.
```powershell
python -m pytest tests -q
```
*Expect exactly:* `19 passed` [VERIFIED]

```powershell
$env:PYTHONPATH="src"
python -c "from veriqa.service import VeriQAService; s=VeriQAService(); r=s.ask('What year did the Island''s flax mills close?'); print(r['answer'], round(r['risk'],4), r['abstained'])"
```
*Expect:* `1965 0.5714 False` [VERIFIED]

**STEP 8 — start the UI**
```powershell
streamlit run app\streamlit_app.py
```
*Expect:* `You can now view your Streamlit app in your browser. URL: http://localhost:8501`,
and a browser tab opening automatically. First load takes ~3–5 s while the 57 MB model loads.

**STEP 9 — ask a question.** Type one from the verified list in
`DEMO_SCREENSHOT_GUIDE.md` and press **Ask**.

**STEP 10 — observe.** You should see the answer (or the abstention box), the supporting
passage with the answer span highlighted, the risk value, per-stage timings, and two
expanders: the 16 reliability features and all retrieved evidence.

**To stop:** `Ctrl+C` in PowerShell.

---

### OPTION B — reproduce the experiments (~15 min, no retraining)

Regenerates every table and figure from the existing predictions.
```powershell
cd "C:\Users\harsh\Claude\Projects\IEEE Paper\VeriQA"
.\.venv\Scripts\Activate.ps1
python experiments\04_train_evaluate.py      # E0 negative control runs FIRST
python experiments\05_ablation.py
python experiments\06_domain_shift.py
python experiments\07_latency.py
python experiments\08_error_analysis.py
python experiments\09_figures_and_freeze.py
```
*Expect from `04`:* `E0 NEGATIVE CONTROL ROC-AUC = 0.4887 ... E0: PASS`, then the six-method
table with `P1 0.6390`. If E0 falls outside [0.45, 0.55] the script **halts by design** and
refuses to print downstream numbers.

⚠️ `09_figures_and_freeze.py` **overwrites `results/RESULT_FREEZE.json`** with a new
timestamp. The numbers should be identical (fixed seed), but back the file up first if you
want to preserve the exact artefact the paper cites.

### OPTION C — full pipeline from scratch (~25–40 min)

Only if you want to rebuild corpora and retrain the reader.
```powershell
python experiments\01_build_corpus.py        # ~2 min, writes models\artifacts_*.pkl
python experiments\02_run_pipeline.py A      # ~8 min, resumable
python experiments\02_run_pipeline.py B      # ~2 min
# then all of Option B
```
`02` is resumable: pass a seconds budget, e.g. `python experiments\02_run_pipeline.py A 120`,
and re-invoke until it reports the full count. Delete
`results\raw\predictions_A.jsonl` first if you want a clean rebuild.

**Use Option A first.**

---

## PART 6 — Runnability verification (what I actually executed)

| Check | Result | Evidence |
|---|---|---|
| `pytest tests -q` | **19 passed** in 7.2 s | [VERIFIED] |
| `VeriQAService()` loads | 3.1 s; 5,637 passages, 11,921 questions, threshold 0.5714 | [VERIFIED] |
| Supported question answers | `"abomasum"`, risk 0.5714, 158.8 ms total | [VERIFIED] |
| Unsupported question abstains | 4/4 abstained, risk 0.89–0.97, with reasons | [VERIFIED] |
| FastAPI `GET /health` | `200 {'status':'ok','frozen':True}` | [VERIFIED] |
| FastAPI `POST /ask` | `200`, answer `'1965'`, risk 0.5714, 5 evidence passages, 16 features | [VERIFIED] |
| FastAPI `GET /evaluate` | `200`, returns frozen headline | [VERIFIED] |
| FastAPI validation | empty question → `422` | [VERIFIED] |
| Streamlit HTTP server | `200` on `/` and `/healthz` | [VERIFIED] |
| Streamlit script executes | `AppTest` ran with **no exception**; 3 tabs, 1 text input, 2 sliders, 1 button, 5 metrics, 5 dataframes | [VERIFIED] |
| Streamlit abstention flow | error box + risk `0.97` vs threshold `0.57` + 2 reasons + suppressed candidate `1998` + passage from *Paris* | [VERIFIED] |

**Nothing in the scientific methodology was modified to make anything run.** No code was
changed during this audit.

### Known issues found during verification

1. **SQLite query logging silently fails on network drives.** `data/veriqa.sqlite` was
   created but `CREATE TABLE` failed with `disk I/O error` on the sandbox's mounted
   filesystem. [VERIFIED] The same code succeeds on local disk. `service.py` wraps logging
   in `try/except`, so **the app never crashes** — logging just does nothing. On your local
   NTFS drive it should work; check with:
   ```powershell
   python -c "import sqlite3;c=sqlite3.connect('data/veriqa.sqlite');print(c.execute('select count(*) from queries').fetchone())"
   ```
2. **The risk score is coarse.** Isotonic calibration is piecewise-constant, and over 150
   sampled questions it emitted only **9 distinct values**: 0.5714, 0.8254, 0.8521, 0.8750,
   0.8898, 0.9447, 0.9556, 0.9706, 1.0. [VERIFIED] **Only 0.5714 sits below the default
   threshold**, so in practice the gate is near-binary and the slider changes decisions
   only when dragged across one of those breakpoints. Expect it to look stepped, not
   smooth. This is a real property of the trained model, not a UI bug.
3. **Streamlit deprecation warnings** about `use_container_width`. Cosmetic; harmless on
   1.61.x.
4. *(Resolved in the final paper.)* **The paper's Fig. 1 mentions document ingestion generally;
   there is no upload UI.** See Discrepancy 2.

---

## PART 10 — Where the paper's numbers come from [VERIFIED live against the files]

| Paper claim | Source file | Field | Produced by | Re-runnable | Frozen |
|---|---|---|---|---|---|
| Base reader accuracy **0.1144** | `results/tables/T1_dataset.csv` | `1 − error_rate[test]` | `04_train_evaluate.py` | yes | ✅ |
| B5 ROC-AUC **0.5242** | `results/tables/T2_main_comparison.csv` | `error_auc[B5]` | `04_train_evaluate.py` | yes | ✅ |
| P1 ROC-AUC **0.6390** | `results/tables/T2_main_comparison.csv` | `error_auc[P1]` | `04_train_evaluate.py` | yes | ✅ |
| P1 AURC **0.8423** | `results/tables/T2_main_comparison.csv` | `aurc[P1]` | `04_train_evaluate.py` | yes | ✅ |
| B5 AURC **0.8785** | `results/tables/T2_main_comparison.csv` | `aurc[B5]` | `04_train_evaluate.py` | yes | ✅ |
| ΔAURC **−0.0345** | `results/raw/E2_p1_vs_b5.json` | `delta_aurc_P1_minus_B5` | `04_train_evaluate.py` | yes | ✅ |
| Reliability latency **1.96 ms** | `results/tables/T7_latency.csv` | `median_ms[RELIABILITY LAYER TOTAL]` | `07_latency.py` | yes | ✅ |

All seven re-read live from the files in this session and matched exactly.

**Reproducibility expectation.** Everything except latency is deterministic under seed
20260814 and should reproduce byte-identically. **Latency (T7) will differ on your
hardware** — that is expected and correct; it is a property of your CPU, not a bug.

---

## PART 11 — What the UI actually shows [VERIFIED by AppTest]

**Layout:** title, caption, and three tabs — **Ask**, **Frozen results**, **About**.

**Ask tab.** Left: a `Question` text input (pre-filled with the ruminants question) and an
`Ask` button. Right: a **Risk tolerance** slider (0.0–1.0, default 0.5714), a **Passages
retrieved (k)** slider (1–10, default 5), and a `Corpus passages` metric showing 5,637.

**After clicking Ask:**
- *If it answers:* a green success box with the answer, then the estimated error
  probability and the threshold.
- *If it abstains:* a red box reading **"I cannot answer this reliably from the provided
  documents."**, the risk and threshold, a **"Why:"** bulleted list of triggered reasons,
  and a collapsed expander **"Show the candidate answer that was suppressed"**.
- Then, in both cases: **"Supporting passage"** with the article name and the passage text,
  with the answer span **highlighted in orange** (via markdown substitution — it highlights
  only when the span occurs verbatim in the passage).
- Three metrics: retrieval / reader / reliability-layer timings in ms.
- Expander: **the 16 reliability features** as a two-column dataframe.
- Expander: **all retrieved evidence** — each passage's article, fused score, first 400 chars.

**Frozen results tab.** Four metrics (P1 AUC 0.639, B5 AUC 0.524, ΔAURC −0.0345,
reliability layer 2.0 ms), the test-set caption, a **pre-registered gates** table showing
4 PASS / 3 FAIL, three figures (risk–coverage, ablation, calibration), and four result
tables (T2, T4, T6, T7).

**About tab.** Static text separating what is ours from what is adapted, and the reader
limitation.

**Confirmed absent:** no document upload, no PDF/TXT/MD ingestion, no side-by-side baseline
pane, no visible query-history panel (logging is DB-only and currently silent), no export.

---

## PART 12 — Dependencies

| Item | Value |
|---|---|
| Python | **3.10** (verified on 3.10.12). 3.11/3.12 will likely work; pins are not tested there. |
| Requirements file | `requirements.txt` |
| Experiment packages | numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.9, rank-bm25 0.2.2, pytest 8.x |
| UI packages | streamlit ≥1.30 (tested 1.61.1), fastapi ≥0.110 (tested 0.141.1), uvicorn, pydantic ≥2 |
| Model files | **None downloaded.** `models/*.pkl` are built from SQuAD by `01_build_corpus.py` |
| Dataset files | `data/raw/{train,dev}-v2.0.json`, 46 MB, already present |
| Environment variables | none required. Set `PYTHONPATH=src` only for ad-hoc `python -c` calls; the scripts and the app set their own path |
| API keys | **none** |
| Ports | Streamlit **8501**; FastAPI **8000** if you run it |
| External services | none |
| Internet | setup only |
| Disk | **~144 MB** total (data 53 MB, models 76 MB, results 8.4 MB, paper 6.2 MB) plus ~400 MB for the venv |
| RAM | **~340 MB** peak for service + one query [VERIFIED]. 2 GB free is comfortable. |
| CPU | any modern x86-64; single-threaded is fine. No GPU. |

**Runs fully offline after `pip install`.** [VERIFIED]

---

## PART 13 — Minimal smoke test

```powershell
cd "C:\Users\harsh\Claude\Projects\IEEE Paper\VeriQA"
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
$env:PYTHONPATH="src"
python -c "from veriqa.service import VeriQAService; s=VeriQAService(); r=s.ask('Where was the band Blue Cheer from?'); print(r['abstained'], r['answer'], round(r['risk'],4))"
```

**Success looks exactly like:**
```
19 passed
False San Francisco 0.5714
```
If you see that, the system is working and you can proceed to the UI.

**If it fails:** `ModuleNotFoundError: veriqa` → you forgot `$env:PYTHONPATH="src"`.
`FileNotFoundError: models/artifacts_A.pkl` → you are in the wrong directory.

---

## PART 16 — Recorded requirement (not acted on)

**The final IEEE paper must be exactly 7 pages including references, and similarity must be
under 3%.** The current PDF is **8 pages**. Both requirements are recorded here and will be
addressed in Phase 5, not now. No manuscript changes were made in this handoff.

---

## FINAL HANDOFF SUMMARY

**A. What was built** — a complete, working, tested CPU-only selective QA system: 2,211
lines of Python, 8 experiment scripts, 19 tests, a Streamlit UI, a FastAPI backend, 10
result tables, 5 figures, and a frozen result file.

**B. What is runnable** — all of it. **[VERIFIED]** tests, service, FastAPI (4 endpoints),
Streamlit (script executes, abstention renders).

**C. Run commands** — Part 4, Option A.

**D. Demo** — `DEMO_SCREENSHOT_GUIDE.md`, with live-verified questions.

**E. Expected output** — Part 13.

**F. Screenshots** — `DEMO_SCREENSHOT_GUIDE.md`.

**G. Result mapping** — Part 10, all seven verified live.

**H. Dependencies** — Part 12.

**I. Known limitations** — classical reader (base accuracy 0.1144), 9-valued risk score,
no document upload, SQuAD-only ingestion, single-machine CPU, English Wikipedia only.

**J. Documented but not implemented** — `experiments/03_extract_features.py` (absent by
design; feature extraction lives in `02`); PDF/TXT/MD ingestion; document upload;
side-by-side baseline pane; visible query history.

**K. Still needs fixing** — (i) verify SQLite logging works on your Windows drive;
(ii) *(done)* correct `docs/ARCHITECTURE.md` and `docs/demo/DEMO_GUIDE.md`, which describe features
that do not exist; (iii) optionally silence the Streamlit deprecation warnings. **None of
these affect the reported results.**

**I am not saying "everything is ready".** I am saying: every component I claim works, I
executed in this session, and the output is quoted above. The reader is weak by
construction and you will see that immediately when you use it — most questions abstain.
That is the honest behaviour of the system as measured, and it matches the paper.
