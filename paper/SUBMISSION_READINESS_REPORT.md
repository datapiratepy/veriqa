# VeriQA — Submission Readiness Report (Final)

**Manuscript:** `VERIQA_IEEE_FINAL.tex` / `.pdf` · 8 pages · official `IEEEtran.cls` v1.8b
**Frozen results:** `../results/RESULT_FREEZE.json` (2026-08-13T09:28:40, seed 20260814)
**Numerical audit:** **86 / 86** claims verified programmatically against frozen artefacts

---

## 1. What changed in this repair pass

### Formatting — the hard blocker

**Official IEEEtran was obtained and used.** `IEEEtran.cls` v1.8b was unavailable in the
build environment (CTAN blocked; no root for `apt`; `tlmgr` unusable). It was recovered
from a GitHub mirror of the official IEEE conference template, verified as v1.8b by
Michael Shell, and installed into `TEXMFHOME`. The manuscript now compiles with
`\documentclass[conference]{IEEEtran}` and the class file itself was **not modified**.

**Every layout defect was fixed at its cause.** Four overfull `\hbox` warnings were traced
to tables whose column widths exceeded the IEEEtran text width — Table IV by 33.5 pt,
Table VI by 13.6 pt, Table X by 3.1 pt — plus one paragraph of unbreakable `\texttt{}`
tokens. Columns were narrowed and the paragraph wrapped in `sloppypar`. Two further tables
sitting within 0.15 cm of the limit were narrowed pre-emptively. **`\resizebox` was not
used anywhere**; all table body text remains at 8 pt.

**Two defects were found by looking at the rendered pages**, not by the compiler:
Figure 2's y-axis label was clipped mid-word inside the PNG ("…among answerec"), and the
references filled only one column of page 8. The figure was regenerated at 300 dpi with a
shorter label; `\IEEEtriggeratref{7}` now balances the bibliography across both columns.

Final state: **0 overfull hbox, 0 overfull vbox, 0 undefined references, 0 words outside
the text block, 0 overlapping word pairs across all 8 pages.** See `VISUAL_QA_REPORT.md`.

### References — the integrity blocker

All three previously incomplete entries are resolved from primary sources:

| Was | Now |
|---|---|
| [4] Joren *et al.* — only first author known | Full list confirmed from the arXiv record: **Joren, Zhang, Ferng, Juan, Taly, Rashtchian** |
| [5] arXiv:2403.01461 — authors unknown | **Abdumalikov (Tartu), Minervini (Edinburgh), Kementchedjhieva (Copenhagen)** |
| [10] PAQ — page range unconfirmed | **TACL vol. 9, pp. 1098–1115**, confirmed via ACL Anthology `2021.tacl-1.65` |

**All 13 references are now fully verified** — authors, title, venue, year, volume/pages
and DOI where one exists. No DOI, page number, author name or venue was invented. A
BibTeX file `VERIQA_IEEE_FINAL.bib` is supplied. Every `\cite` resolves and every
`\bibitem` is cited — 13 citations, 13 entries, no orphans.

### Literature — 2025–26 coverage

**Added:** Muhamed *et al.*, *RefusalBench* (arXiv:2510.10390, Oct 2025) — verified via
arXiv, Microsoft Research and NeurIPS 2025 listings. Cited for two specific findings
relevant to us: refusal accuracy of frontier models falls below 50% on multi-document
tasks, and refusal decomposes into separable detection and categorisation skills. It
overlaps our topic (selective refusal in grounded models) but not our contribution: it
evaluates large generative models by perturbation, whereas we measure the marginal value
of retrieval features over a reader-side calibrator in a CPU-only extractive pipeline.

**Already present and load-bearing:** Joren *et al.* (Sufficient Context, ICLR 2025),
Abdumalikov *et al.* (answerability under retrieval, 2024), Dhuliawala *et al.* (reading
calibration at scale, ACL Findings 2022), Lewis *et al.* (PAQ/RePAQ selective QA, TACL 2021).

**Still excluded.** SURE-RAG (arXiv:2605.03534), *Evidence-Calibrated RAG* (JTIE),
*Uncertainty Quantification for Retrieval-Augmented Reasoning* (arXiv:2510.11483) and
*QA-Calibration* (ICLR 2025) could not be verified to citation standard — author lists
and venue records did not resolve. Per the project's integrity rules they are **not**
cited. Identifiers are recorded here so the team can verify and add them; each would
strengthen Section II. **Padding the bibliography with unverified entries was rejected.**

### Language and claims

- `improves AURC` → **`reduces AURC`** throughout, with "lower is better" stated in §II
  and in every relevant caption. Verified absent by scan.
- No "statistically significant" anywhere except an explicit disclaimer that no
  null-hypothesis test was run.
- Domain transfer: "the point estimate shows no large degradation", never "no degradation".
- Calibration: isotonic "calibrates the mapping from risk score to observed error
  frequency; it confers no guarantee", plus the explicit statement that the system "must
  not be described as delivering any guaranteed accuracy level".
- Banned-phrase scan clean: no *state-of-the-art*, *revolutionary*, *groundbreaking*,
  *first-ever*, *novel framework*.
- *"We claim the measurement, not the mechanism"* appears in the abstract and §II.

### Results

**Unchanged.** No experimental number was altered. 86/86 claims re-verified against the
frozen artefacts after the rewrite. All three failed gates (G1, G2, G5) remain reported in
the abstract, §IX-D, Table X and §XII.

---

## 2. Final reviewer attack

### Reviewer 1 — NLP/QA

| Question | Answer |
|---|---|
| Novelty defensible? | Yes, as framed. The claim is an incremental measurement, not a mechanism; [1] is credited and reimplemented as B5. |
| Closest prior work cited? | Yes — [1], [3], [4], [5], [6], [7] span calibration, sufficiency, answerability, refusal and retrieval-side selective QA. |
| Citations accurate? | All 13 verified against primary sources. |
| Reproducible? | Seed, single split function, checksums, 19 tests, frozen file, per-number provenance. |
| Baselines fair? | B5 uses the same model class, hyperparameters, splits and calibration as P1; only features differ. |
| Statistics valid? | Paired bootstrap on identical article resamples; no significance claimed. |
| **Rejection risk** | **Base accuracy 0.1144 and a classical reader.** Stated, not concealed. Unfixable without a transformer. |

### Reviewer 2 — IR/RAG

| Question | Answer |
|---|---|
| Answerability handled? | §IV-A with a measured contamination audit (14.89% → 2.39%) and the sibling-masking protocol derived from it. Strongest methodological section. |
| Retriever competitive? | No. TF-IDF+SVD with 89.1% top-5 gold recall; no dense-neural comparison. Conceded in §XII. |
| Recent prior art? | Now covers 2021–2025 including RefusalBench and Sufficient Context. |
| Scale? | 5,637 passages, exact index; million-scale untested. Conceded. |
| **Rejection risk** | **No neural-retriever baseline.** |

### Reviewer 3 — IEEE systems

| Question | Answer |
|---|---|
| Professionally typeset? | Official IEEEtran v1.8b, 0 overfull boxes, every page visually inspected. |
| Reject on formatting? | **No.** This was the previous blocker and is now cleared. |
| Implies production readiness? | No — explicitly denied in §XII. |
| Implementation accurately described? | Yes — §VI states no transformer, neural embedding or pretrained LM is used anywhere. |
| **Rejection risk** | **No deployment or user evidence.** |

**Net:** appropriate for a student research track, reproducibility/negative-results track,
or a regional IEEE conference. Not competitive at ACL/EMNLP/SIGIR main tracks — because of
reader and retriever strength, both stated rather than hidden.

---

## 3. Remaining human actions

**Blocking**

1. **Confirm the target venue's page limit.** The paper is 8 pages; many IEEE conferences
   cap camera-ready at 6 pages plus references.
2. **Confirm institutional affiliation string**, and add e-mail addresses / ORCIDs if the
   venue requires them. **None were invented.**
3. **Confirm spelling of all five names**, and that Divya Tyagi (guide) and Snigdha Kesh
   (coordinator) consent to authorship and accept the role footnote.
4. **Run the institution-approved similarity and AI-detection check.** No such tool is
   available in this environment and no percentage can be estimated from here.
5. **All five authors read the manuscript end to end.**

**Recommended**

6. Re-run the experiment with a transformer reader on a machine with model-repository
   access. Highest research value; changes every absolute number and may narrow the
   conclusion.
7. Verify and cite the four excluded 2025–26 papers.
8. Compile once on your own machine to confirm the fonts and figures resolve there.

**Do not claim** publication, acceptance, peer review, state-of-the-art performance,
production readiness, or superiority over any commercial system.

---

## SUBMISSION READINESS SCORECARD

| Dimension | Score | Basis |
|---|---|---|
| **Technical correctness** | **9/10** | 86/86 claims traced to frozen artefacts; 19 tests pass; negative control passes; a metric-polarity bug was caught and fixed pre-freeze. −1: no independent re-implementation. |
| **Novelty positioning** | **8/10** | Framed as incremental measurement; closest prior work cited and credited; B5 is a faithful reimplementation of [1]. −2: the gap is narrow; a reviewer may still call it incremental. |
| **Experimental rigour** | **8/10** | Pre-registered gates unchanged, article-level splits, negative control, paired cluster bootstrap, ablation + permutation cross-check, threshold sensitivity, transfer test. −2: weak reader, one transfer setting, no neural-retriever comparison. |
| **Literature coverage** | **8/10** | 13 fully verified references spanning 1990–2025, including two 2024–25 works on answerability and selective refusal. −2: four relevant 2025–26 papers identified but unverifiable, so excluded. |
| **IEEE formatting readiness** | **10/10** | Official IEEEtran v1.8b; 0 overfull boxes; 0 words outside the text block; 0 overlaps; all 8 pages visually inspected; two real defects found and fixed. |
| **Reproducibility** | **10/10** | Seed, single split function, data checksums, config and table hashes, pinned requirements, resumable scripts, 19 tests, frozen result file. |
| **Overall submission readiness** | **9/10** | Substantively and typographically ready. Remaining items are administrative, not technical. |

### Remaining blockers

**No technical or integrity blockers remain.** The three that existed at the previous audit
— incomplete references, no official IEEEtran compilation, and PDF layout defects — are all
cleared.

Four administrative items stand between this and submission: venue page limit, affiliation
and contact details, authorship consent from the guide and coordinator, and the
institutional similarity check. None requires further engineering.

The paper's honest weaknesses — base reader accuracy 0.1144, no method reaching AUC 0.70,
and an unreachable 80% accuracy target — are reported in the abstract, results, gate table
and limitations. They will attract reviewer criticism. Concealing them would attract worse.
