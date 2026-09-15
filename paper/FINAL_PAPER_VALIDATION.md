# VeriQA — Final Paper Validation

**Source:** `paper/VERIQA_IEEE_FINAL.tex`
**PDF:** `paper/VERIQA_IEEE_FINAL.pdf`
**Bibliography:** `paper/VERIQA_IEEE_FINAL.bib`
**Class:** official `IEEEtran.cls` v1.8b (Michael Shell, 2015/08/26), `[conference]`
**Validated:** 17 August 2026 · **Revision 2** (two wording corrections, revalidated)

---

## 0. Revision 2 — two surgical wording corrections

Applied to `VERIQA_IEEE_FINAL.tex` only. **No experiment, number, table, figure, citation,
bibliography entry, author field or scientific conclusion was altered**, and nothing in
`src/`, `experiments/`, `results/` or `models/` was touched (verified: 0 non-bytecode files
modified; `RESULT_FREEZE.json` md5 `ac7e359d…` and `T2_main_comparison.csv` md5
`c556b9f0…` unchanged).

**Correction 1 — abstract ΔAURC sign.** The abstract previously gave the interval unsigned
as "(paired interval 0.0191--0.0495, excluding zero)", which was ambiguous against the
body's signed definition ΔAURC = P1 − B5 = −0.0345. The sentence now reads, as rendered:

> "…and reduces the area under the risk–coverage curve by 0.0345 (paired 95% bootstrap
> interval for ΔAURC = [−0.0495, −0.0191], excluding zero)…"

The magnitude of the reduction (0.0345) and the signed interval ([−0.0495, −0.0191]) now
agree with §VII-C and with `results/raw/E2_p1_vs_b5.json`. "lowers" was also changed to
"reduces" for consistency with the lower-is-better convention stated in §II. **The
underlying result is unchanged.**

**Correction 2 — Fig. 2 caption softened.** "Lower curves are better at every coverage
level" overstated a universal ordering. The caption now reads:

> "Lower curves indicate lower selective risk; P1 generally lies below the baselines across
> the coverage range."

The figure, its data, plotted lines, axis labels and legend are untouched.

**Post-edit revalidation:** page count still **exactly 7**; 0 overfull `\hbox`, 0 overfull
`\vbox`, 0 undefined references, 0 missing files; 0 words outside the text box and 0
overlapping word pairs across all 7 pages; vertical fill 96% on pages 1–6 and 93% on page 7;
13 citations / 13 bibitems with none uncited or undefined; both embedded figures still
251.1 pt wide (exactly one column) on pages 4 and 5. All 13 locked headline values
re-checked against the frozen artefacts — **13/13 match**. Pages 1, 4 and 7 were re-rendered
and visually inspected: the abstract rewraps cleanly with no orphan line, the Fig. 2 caption
sets on three lines without overflow, and page 7 keeps its balanced two-column
bibliography. Current renders: `paper/qa8/g-1..7.png`.

---

## 1. Page count — the hard requirement

| | |
|---|---|
| **Rendered page count** | **7** |
| Method | `pdfinfo VERIQA_IEEE_FINAL.pdf` on the compiled artefact, not estimated from source |
| Page size | 612 × 792 pt (US Letter) |
| References included in the 7 pages? | **Yes** — all 13 entries sit on page 7 |
| Font shrinking used? | **No.** Body 10 pt, tables 8 pt, captions at IEEEtran defaults |
| `\resizebox`, margin changes, or spacing hacks? | **None** |

**How one page was removed without cutting science.** The 8-page version contained the
ablation both as a bar chart *and* as Table VI — the same data twice. The figure was
dropped and the table kept, since the locked ablation values need exact intervals that a
chart cannot convey. The two-row contamination table and the three-row threshold-sensitivity
table were converted to prose, which is more compact at that size. Prose throughout was
rewritten more tightly. The calibration figure, previously absent, was **added**. Net float
change: 10 tables → 7 tables, 3 figures → 3 figures.

## 2. Build status

| Check | Result |
|---|---|
| Overfull `\hbox` | **0** |
| Overfull `\vbox` | **0** |
| Undefined references or citations | **0** |
| Missing figures | **0** |
| Compiler | pdfTeX 3.141592653-2.6-1.40.22, two passes |

## 3. Geometry and layout (programmatic, `pdfplumber`)

Text block is a consistent x ∈ [49.0, 563.0] pt on every page — IEEEtran letter geometry.

| Page | Words | Outside text box | Overlapping word pairs | Vertical fill |
|---|---|---|---|---|
| 1 | 568 | 0 | 0 | 96% |
| 2 | 721 | 0 | 0 | 96% |
| 3 | 725 | 0 | 0 | 96% |
| 4 | 501 | 0 | 0 | 96% |
| 5 | 475 | 0 | 0 | 96% |
| 6 | 598 | 0 | 0 | 96% |
| 7 | 340 | 0 | 0 | 93% |
| | | **0 total** | **0 total** | — |

**No page is blank or nearly blank.** Page 7 reaches 93% vertical fill, carrying two tables,
the conclusion, the acknowledgment and the complete bibliography.

## 4. Page-by-page visual inspection

All seven pages were rendered to PNG at 110 dpi (`paper/qa7/f-*.png`) and examined.

| Page | Contents | Finding |
|---|---|---|
| 1 | Title, five authors, affiliation, role footnote, abstract, index terms, Introduction, contributions | Title sets on three centred lines; all five names on one line in the required order; no clipping ✅ |
| 2 | Contributions (cont.), Related Work, Problem Formulation, Data §IV-A | Citations [1]–[13] render; inline math correct; "We claim the measurement, not the mechanism" present ✅ |
| 3 | Fig. 1 (pipeline), collections and partitions, pre-registration, §V Retrieval and Reader | Framed pipeline figure fits its column; `\texttt{}` identifiers break cleanly ✅ |
| 4 | Tables I–II, Fig. 2 (risk–coverage), §VI Baselines, §VII-A–C Results | Two tables stack in column 1, figure in column 2, no collision ✅ |
| 5 | Table III (main comparison, full width), Fig. 3 (calibration), §VII-D–E, §VIII | Seven-column table spans both columns cleanly; all cells readable ✅ |
| 6 | Table IV (ablation, full width), Table V (taxonomy), §X Limitations, §XI Conclusion | Ablation table readable including negative interval `−0.0044–0.0189` ✅ |
| 7 | Table VI (transfer, full width), Table VII (cost), conclusion tail, acknowledgment, 13 references | References balanced across both columns via `\IEEEtriggeratref{7}`; DOIs and URLs unbroken ✅ |

**Explicit conformance:** no overlapping text · no clipped text · nothing outside page or
column boundaries · no overlapping tables or figures · all tables readable at 8 pt · both
plotted figures readable with 7–8 pt axis text · author block readable · references readable.

## 5. Result consistency — **127 / 127 verified**

Every numeric claim in the source was checked programmatically against the frozen
artefacts in `results/`. **All 127 matched.** This covers all six procedures' ROC-AUC and
AURC with intervals, both selective-accuracy columns, both ΔAURC comparisons with
intervals, ECE before and after, the calibration operating point, all seven ablation rows
with intervals, six permutation-importance values, all six transfer rows, every latency
cell, the resource figures, dataset and partition counts, the five taxonomy rows, the
threshold-sensitivity values, and the five question-type rows.

Spot check of the locked headline values:

| Claim | Value in paper | Source artefact |
|---|---|---|
| Base reader accuracy | 0.1144 | `results/tables/T1_dataset.csv` |
| B5 error-prediction ROC-AUC | 0.5242 | `results/tables/T2_main_comparison.csv` |
| P1 error-prediction ROC-AUC | 0.6390 [0.6127, 0.6668] | `results/tables/T2_main_comparison.csv` |
| P1 AURC | 0.8423 [0.8029, 0.8731] | `results/tables/T2_main_comparison.csv` |
| B5 AURC | 0.8785 | `results/tables/T2_main_comparison.csv` |
| ΔAURC (P1 − B5) | −0.0345 [−0.0495, −0.0191] | `results/raw/E2_p1_vs_b5.json` |
| Reliability-layer overhead | 1.96 ms median, 2.37 ms p95 | `results/tables/T7_latency.csv` |
| End-to-end query | 63.77 ms median, 83.03 ms p95 | `results/tables/T7_latency.csv` |
| Negative control | 0.4887 in [0.45, 0.55] | `results/raw/E0_negative_control.json` |

**No experimental value was changed, recomputed or re-run for this revision.**

## 6. Claim discipline

| Term | Occurrences | Status |
|---|---|---|
| state-of-the-art, revolutionary, groundbreaking, first-ever, novel framework, hallucination-free, human-level, highly accurate | **0** | clean |
| "improves AURC" | **0** | replaced by "reduces AURC"; lower-is-better stated in §II |
| "statistically significant" | 1 | **negated:** "we do not describe the outcome as statistically significant" |
| "production-ready" | 1 | **negated:** "it is not claimed to be production-ready" |
| "transformer reader" | 1 | **future work:** "Repeating the measurement with a transformer reader is the most consequential outstanding step" |

Explicitly retained: `0.6390` and `0.5242` are labelled *error-prediction ROC-AUC*, never
accuracy · `−0.0345` is labelled the AURC difference P1 − B5 · `0.1144` is labelled base
reader accuracy · **4 PASS / 3 FAIL** is stated in §IX with all three failures named ·
"No transformer, learned embedding or pretrained language model appears anywhere in the
evaluated system" · "We claim the measurement, not the mechanism" appears in the abstract
and §II.

## 7. Bibliography

13 references, **13 cited, 0 uncited, 0 undefined.** All were verified against primary
sources in the previous audit: ACL Anthology for [1]–[3], arXiv for [4]–[6], ACL Anthology
and MIT Press for [7], JMLR for [8], NeurIPS proceedings for [9], PMLR for [10], ACM DL for
[11], Wiley for [12], nowpublishers/ACM for [13]. No DOI, page range, author name or venue
was invented.

## 8. Originality check — what was and was not measured

⚠️ **No plagiarism percentage was computed. Turnitin, iThenticate and equivalent services
are unavailable in this environment, so no similarity figure can be reported and none is
claimed.**

What I *could* run is a genuine verbatim-overlap test against the four prior-work texts I
retrieved during this project — the most realistic source of accidental copying. The
rendered PDF text (5,571 words) was compared against those abstracts:

| Source | Shared 7-grams | Shared 8-grams | Shared 10-grams |
|---|---|---|---|
| Joren *et al.* (arXiv 2411.06037) | 0 | 0 | 0 |
| Abdumalikov *et al.* (arXiv 2403.01461) | 0 | 0 | 0 |
| Lewis *et al.* PAQ (TACL 2021) | 0 | 0 | 0 |
| Muhamed *et al.* RefusalBench (arXiv 2510.10390) | 0 | 0 | 0 |

**Longest shared contiguous word run: 4 words — "on retrieval augmented generation"**, an
unavoidable technical term.

Manual audit alongside it: the abstract, introduction, related work, method, results and
conclusion were written from the project's own artefacts, not adapted from any source. All
borrowed ideas remain cited — the trained error-predictor concept to [1], isotonic
calibration to [11], BM25 to [13], LSA to [12], selective prediction to [8], [9]. Standard
terminology (ROC-AUC, AURC, ECE, BM25, TF-IDF, SVD, SQuAD 2.0, isotonic regression) is
retained deliberately; substituting it would harm clarity and is not what similarity
checking targets. **No text was altered to evade a checker and no citation was weakened to
reduce overlap.**

**Residual risk:** the true similarity score depends on the checker's reference corpus,
which I cannot see. Method-section phrasing describing standard techniques is where any
score will concentrate. The team must run the institution-approved tool.

## 9. Unresolved issues

**None blocking.** Two administrative items remain, unchanged from the previous audit and
outside my control:

1. Institutional e-mail addresses and ORCID identifiers are omitted — **none were
   invented.** Add them if the venue requires.
2. Authorship consent from D. Tyagi (guide) and S. Kesh (coordinator) for the role
   footnote, and confirmation of name spellings.

Also note: the venue's page limit should be confirmed against 7 pages, and the
institution-approved similarity check must still be run (§8).

## 10. File status

**Canonical:** `VERIQA_IEEE_FINAL.tex` · `VERIQA_IEEE_FINAL.pdf` (7 pages) ·
`VERIQA_IEEE_FINAL.bib` · `figures/` · this file · `VISUAL_QA_REPORT.md` ·
`SUBMISSION_READINESS_REPORT.md` · `README_SUBMISSION.md`

**Superseded, retained for traceability only:** `VERIQA_IEEE_MANUSCRIPT.{md,tex}`,
`VERIQA_MANUSCRIPT_LAYOUT_PREVIEW.pdf`, `pg-*.png`, `qa/v2-*.png` (8-page renders).
Current renders are `qa7/f-1..7.png`. No file is named FINAL2, FINAL_NEW or similar.

**Nothing in `src/`, `experiments/`, `results/` or `models/` was modified during this
paper-production task.**
