# VeriQA — Final Project Audit

**Date:** 13 August 2026 · **Results frozen:** see `results/RESULT_FREEZE.json`

---

## 1. What was built

A complete, tested, runnable CPU-only selective extractive QA system:

- **Ingestion** — SQuAD 2.0 loader, corpus construction with stratified per-article question
  sampling, checksum verification.
- **Retrieval** — TF-IDF → truncated SVD (256 dims) exact inner-product index, BM25Okapi, min-max
  fusion, and per-question sibling masking implementing protocol D-002.
- **Reader** — two-stage feature-based extractive reader: sentence ranking, candidate filtering,
  a gradient-boosted span scorer over 20 vectorised span features, and a separately trained
  null-answer head.
- **Reliability layer** — 16 features in three families, gradient-boosted risk model, isotonic
  calibration, threshold selection on calibration data only.
- **Baselines** — B1–B5 including a faithful reimplementation of the Kamath et al. calibrator.
- **Evaluation** — ROC-AUC, AURC, ECE, selective accuracy, coverage-at-target, SQuAD EM/F1,
  article-clustered bootstrap, paired bootstrap.
- **Application** — FastAPI backend (`/health`, `/ask`, `/evaluate`), Streamlit UI with evidence
  display, span highlighting, risk slider, abstention reasons and a frozen-results tab, SQLite
  query log.
- **Tests** — 19, including hand-computed metric fixtures, feature range/finiteness checks,
  retrieval ordering, sibling masking, and article-disjointness of all four partitions.
- **9 experiment scripts**, resumable, seeded, producing 10 result tables and 5 figures.

## 2. What was experimentally demonstrated

E0 negative control; E1 six-method comparison; E2 paired P1-vs-B5 test; E3 risk–coverage curves;
E4 calibration before/after isotonic with target realisation; E5 seven-configuration ablation plus
permutation importance; E6 transfer to a disjoint corpus; E7 per-stage latency and memory;
E8 error taxonomy, question-type breakdown, correctness-threshold sensitivity and real examples.

All on 3,585 test questions from 36 articles never seen by either model.

## 3. Main research result

**Retrieval-side signals improve error prediction beyond reader-side signals.**
ROC-AUC 0.5242 → 0.6390 [0.6127, 0.6668]. ΔAURC −0.0345 [−0.0495, −0.0191], paired bootstrap over
articles, interval excludes zero. Robust at correctness thresholds 0.4, 0.5 and 0.6. Survives
transfer to a disjoint corpus (0.6494). Costs 1.96 ms.

Ablation localises it: retrieval ΔAURC 0.0274 [0.0149, 0.0411]; agreement 0.0103 [0.0043, 0.0173];
**reader 0.0067 [−0.0044, 0.0189] — interval includes zero.** Permutation importance agrees.

## 4. Did the central hypothesis pass?

**Yes.** Gate G3 — the pre-registered central test — passed, as did G4, G0 and G6.
**Gates G1, G2 and G5 failed.** Four of seven. The comparative claim holds; the absolute
performance claims do not, and the paper leads with both.

## 5–8. Limitations

**Method.** The reader is a classical feature-based span scorer, not a transformer, because no
pretrained weights were reachable (D-009). Mean answer F1 0.145, base accuracy 0.114. Every
result is conditional on this. Correlated feature families mean leave-one-out bounds contribution
from below rather than partitioning credit.

**Dataset.** 2.39% residual unanswerable-label risk after sibling masking (measured, down from
14.89%). SQuAD 2.0's unanswerable questions are adversarially authored and may not resemble
naturally occurring ones. English Wikipedia only. Paragraph-level retrieval, so results do not
transfer directly to systems chunking long documents.

**Computational.** Single-machine CPU. Corpus of 5,637 passages; behaviour at millions of
passages with an approximate index is untested. Latency figures are host-specific.

**Generalisation.** One transfer test only, and it moves both corpus and reader instance without
separating them. No non-English or technical-domain evaluation. No deployment evidence.

## 9. Reproducibility

Seed 20260814 throughout. Data SHA-256 in `data/raw/CHECKSUMS.txt`; config and table hashes in
`RESULT_FREEZE.json`. All splits derive from one seeded function. Experiments are resumable and
write intermediate artefacts. 19 tests pass. Only latency varies across hosts.

## 10. IEEE-paper readiness

`paper/VERIQA_IEEE_MANUSCRIPT.md` and `.tex`, ~5,370 words, 6 tables, 13 sections plus references.

- **Numerical audit:** 65 of 65 claimed values verified programmatically against frozen files.
- **Overclaim scan:** clean — no "state-of-the-art", "novel framework", "revolutionary",
  "groundbreaking", "first-ever" or unsupported "guarantee".
- **Claim separation:** Section IX splits implemented / adapted / engineering contribution /
  experimental contribution / unproven.
- **Layout check:** the LaTeX body compiles to 7 pages under a stand-in two-column class.
  ⚠️ **`IEEEtran.cls` was not available, so IEEE template compliance is NOT verified and must
  not be claimed.**
- ⚠️ **References [2]–[10] are unverified** and must be opened and checked by a human. [1]
  (Kamath, Jia & Liang, ACL 2020) was confirmed against the ACL Anthology.
- Nothing is published, submitted or peer reviewed.

## 11. Remaining manual tasks

1. **Verify references [2]–[10]** — authors, title, venue, year, pages, DOI. Non-negotiable.
2. **Compile the `.tex` against the real IEEEtran template**, then check page count and captions.
3. **Re-run with a transformer reader** on a machine with Hugging Face access. Highest research
   value; changes every absolute number.
4. **Rehearse the demo three times**, once with wifi off.
5. **Practise the viva**, each member on their own layer.
6. **Read the manuscript end to end** — you are the authors and must stand behind every sentence.
7. Choose a venue only if you decide to submit.

## 12. Honest bottom line

The project is complete and defensible. The engineering is real, the experiment is real, the
central hypothesis was tested and passed, and the failures are reported rather than buried. The
one thing that would materially strengthen it is a transformer reader, and the codebase was
written so that swapping one in requires changing a single module.
