# VeriQA paper

**Quantifying the Incremental Value of Retrieval-Side Reliability Signals for Selective
Extractive Question Answering**

Harsh Kamat, Harsh Ranjan, Manish Kumar, Divya Tyagi, Snigdha Kesh ·
Department of Computer Science and Engineering, AMC Engineering College (VTU), Bengaluru

**Status:** IEEE conference-format manuscript. It is **not published and has not been peer
reviewed**.

## Files

| File | Role |
|---|---|
| [`VERIQA_IEEE_FINAL.pdf`](VERIQA_IEEE_FINAL.pdf) | The compiled paper: 7 pages, US Letter |
| [`VERIQA_IEEE_FINAL.tex`](VERIQA_IEEE_FINAL.tex) | The manuscript source (`IEEEtran`, `[conference]`) |
| [`VERIQA_IEEE_FINAL.bib`](VERIQA_IEEE_FINAL.bib) | BibTeX for all 13 references (the `.tex` uses an inline `thebibliography`; the `.bib` is for venues that require one) |
| [`figures/`](figures/) | `F3_risk_coverage.png` (Fig. 2) and `F4_calibration.png` (Fig. 3); Fig. 1 is drawn in LaTeX |
| [`FINAL_PAPER_VALIDATION.md`](FINAL_PAPER_VALIDATION.md) | Checks on the final revision: build, layout, numerical consistency against the frozen results, claim discipline, bibliography |
| [`SUBMISSION_READINESS_REPORT.md`](SUBMISSION_READINESS_REPORT.md) | Earlier audit, written for the 8-page revision: reference ledger, reviewer-style critique, remaining actions |

**Compile:** `pdflatex VERIQA_IEEE_FINAL.tex` (run it twice). Requires `IEEEtran.cls` v1.8b.

## Provenance

Every number in the paper traces to `../results/tables/*.csv` and
`../results/RESULT_FREEZE.json` (frozen 2026-08-13T09:28:40, seed `20260814`). Do not
edit numbers by hand: re-run the experiment scripts and re-freeze instead.

Superseded drafts (`VERIQA_IEEE_MANUSCRIPT.md` / `.tex`, a layout-preview PDF built with a
stand-in document class) and the page renders used during layout checks were removed from
the working tree in September 2026. The validation records above still mention them; they
remain available in the repository history.
