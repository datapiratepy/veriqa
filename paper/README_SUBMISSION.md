# VeriQA — Submission Package

**Compile:** `pdflatex VERIQA_IEEE_FINAL.tex` (twice). Requires `IEEEtran.cls` v1.8b.

## Canonical files — use these

| File | Role |
|---|---|
| **`VERIQA_IEEE_FINAL.tex`** | **The manuscript. Official IEEEtran, `[conference]`.** |
| **`VERIQA_IEEE_FINAL.pdf`** | **The compiled paper. 8 pages, letter.** |
| **`VERIQA_IEEE_FINAL.bib`** | BibTeX source for all 13 references (the `.tex` uses an inline `thebibliography`; the `.bib` is provided for venues that require it). |
| `figures/` | `F3_risk_coverage.png`, `F5_ablation.png` (Figs. 2 and 3). Fig. 1 is drawn in LaTeX. |
| `SUBMISSION_READINESS_REPORT.md` | Audit, reviewer responses, reference ledger, remaining actions. |
| `VISUAL_QA_REPORT.md` | Page-by-page layout inspection. |
| `qa/v2-*.png` | The 8 rendered pages used for that inspection. |

## Superseded — do not submit

`VERIQA_IEEE_MANUSCRIPT.md` (Markdown working draft, content-identical),
`VERIQA_IEEE_MANUSCRIPT.tex` (auto-converted, superseded by the hand-built final),
`VERIQA_MANUSCRIPT_LAYOUT_PREVIEW.pdf` (stand-in-class preview from before IEEEtran
was available), `pg-*.png` (renders of that superseded preview).

These are kept for traceability only. **Nothing named `FINAL2`, `FINAL_NEW` or
similar exists; `VERIQA_IEEE_FINAL.*` is the single current version.**

## Provenance

Every number traces to `../results/tables/*.csv` and `../results/RESULT_FREEZE.json`
(frozen 2026-08-13T09:28:40, seed 20260814). An automated audit verifies 86 of 86
numerical claims. Do not edit numbers by hand; re-run the experiment scripts and
re-freeze instead.
