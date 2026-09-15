# VeriQA — Page-by-Page Visual QA Report

**PDF audited:** `VERIQA_IEEE_FINAL.pdf` · 8 pages · 612 × 792 pt (US Letter)
**Class:** official `IEEEtran.cls` **v1.8b** (2015/08/26, Michael Shell), `[conference]`
**Method:** (1) compiler-warning inspection, (2) programmatic bounding-box extraction with
`pdfplumber`, (3) rendering all 8 pages to PNG at 110 dpi and inspecting each visually.

---

## 1. Compiler diagnostics

| Check | Result |
|---|---|
| Overfull `\hbox` | **0** |
| Overfull `\vbox` | **0** |
| Undefined references / citations | **0** |
| Missing figures | **0** |
| Class actually used | `IEEEtran.cls` v1.8b, loaded from `TEXMFHOME` |

Four overfull `\hbox` warnings existed in the first IEEEtran build and were **fixed at
their cause**, not suppressed:

| Cause | Fix |
|---|---|
| Table IV columns summed to 19.37 cm against a 18.19 cm text width (over by 33.5 pt) | Column widths reduced 16.4 → 14.8 cm |
| Table VI over by 13.6 pt | Column widths reduced 15.7 → 13.8 cm |
| Table X over by 3.1 pt | Column widths reduced 7.3 → 6.7 cm |
| Permutation-importance paragraph: unbreakable `\texttt{}` feature names | Wrapped in `sloppypar` |

Tables V and VIII were also narrowed pre-emptively (18.07 → 16.87 cm and 18.17 → 16.67 cm)
because both sat within 0.15 cm of the limit.

## 2. Programmatic geometry audit

Text block measured as a consistent x ∈ [49.0, 563.0], y ∈ [50.9, 718.6] pt on every
body page — exactly IEEEtran's letter geometry.

| Page | Words | x-range | y-range | Outside text box | Overlapping word pairs |
|---|---|---|---|---|---|
| 1 | 570 | [49.0, 563.0] | [52.4, 718.6] | **0** | **0** |
| 2 | 682 | [49.0, 563.0] | [50.9, 718.6] | **0** | **0** |
| 3 | 670 | [49.0, 563.0] | [52.7, 718.6] | **0** | **0** |
| 4 | 546 | [49.0, 563.0] | [52.7, 718.6] | **0** | **0** |
| 5 | 414 | [49.0, 563.0] | [52.7, 718.6] | **0** | **0** |
| 6 | 511 | [49.0, 563.0] | [52.7, 718.6] | **0** | **0** |
| 7 | 481 | [49.0, 563.0] | [52.7, 718.6] | **0** | **0** |
| 8 | 206 | [52.9, 563.0] | [51.1, 272.7] | **0** | **0** |
| | | | **TOTAL** | **0** | **0** |

Embedded images (p5): both at x ∈ [312.0, 563.0], width 251.1 pt = exactly one column,
sitting inside column 2. Neither crosses the gutter.

Nine tokens cross the inter-column gap; all were inspected individually and all are
legitimate full-width elements — title words on p1, the affiliation line, and the
`table*` captions "TABLE IV/V/VI/VIII" on pp. 5–7.

Smallest font: 6.4 pt, occurring only in IEEEtran's own small-caps table captions (the
class's default caption style, not a setting of ours). All table *body* text is 8 pt.

## 3. Page-by-page visual inspection

Each page was rendered and examined. Findings below are from looking at the images, not
from the compiler.

**Page 1 — title, authors, abstract, index terms, Introduction.**
Title sets on three centred lines, no hyphenation break. All five authors on a single
line in the required order, followed by the three-line affiliation. Role footnote sits at
the bottom of column 1, fully inside the margin. Abstract justified, no rivers, no clipped
words. ✅

**Page 2 — contributions list, Related Work, Research Question, Dataset §IV-A.**
Numbered list items align correctly. Inline math (`R ⊆ C`, `g(q,R,â) → [0,1]`, `τ`) renders
correctly. `\texttt{plausible\_answer}` breaks cleanly at the column edge. ✅

**Page 3 — Table I, Figure 1, corpora and splits.**
Table I sits at the top of column 1 with its caption above, inside the column. Figure 1
(the pipeline box) occupies the top of column 2; the framed minipage does not touch the
column boundary, and every monospaced line fits without wrapping. Heading "V. System
Architecture" clears the figure. ✅

**Page 4 — Tables II and III side by side, §VII, §VIII, §IX-A–C.**
Table II (16 features, heavy text wrapping) in column 1 and Table III in column 2 — both
fully readable at 8 pt, no cell clipping, no collision between them. Bullet list of ΔAURC
results renders with correct math. ✅

**Page 5 — Table IV (`table*`), Figures 2 and 3, §IX-D–E.**
Table IV spans both columns at the top, all seven columns readable, no content touching
either edge. Figures 2 and 3 stack in column 2 at exactly column width with captions
below. Body text flows around them without collision. ✅
*Defect found and fixed here:* Figure 2's y-axis label was clipped mid-word
("…among answerec"). The figure was regenerated with a shorter label ("selective risk")
and tight bounding box; it now renders complete.

**Page 6 — Tables V, VI (`table*`), Table VII, §X.**
Tables V and VI stack at the top spanning both columns with clear separation. Table VII
sits in column 1 below them. The 0.5-threshold row in Table V wraps its interval onto two
lines, which is legible and correctly aligned. Verbatim example answers render with proper
quotation marks and the accented "Brasília". ✅

**Page 7 — Table VIII (`table*`), Tables IX and X, §XI, §XII.**
Table VIII spans both columns; Tables IX and X stack in column 1, each fully inside the
column. Gate table shows all seven gates with PASS/FAIL. The `\pm3` and `\in[0.45,0.55]`
math renders correctly inside table cells. ✅

**Page 8 — Acknowledgment and References.**
*Defect found and fixed here:* references originally filled only column 1, leaving a badly
imbalanced page. `\IEEEtriggeratref{7}` was added, and the 13 references now split evenly
across both columns ([1]–[6] left, [7]–[13] right). All entries readable, no margin
violations, DOIs and URLs unbroken. ✅

## 4. Explicit conformance statement

Based on the inspection above, the final PDF has:

- ✅ **no overlapping text** — 0 overlapping word pairs across all 8 pages
- ✅ **no clipped text** — 0 words truncated; the one clipped figure label was fixed
- ✅ **no text outside page boundaries** — 0 words beyond x ∈ [49, 563], y ∈ [50, 745]
- ✅ **no text outside column boundaries** — the only cross-gutter tokens are the title,
  affiliation and `table*` captions, which are full-width by design
- ✅ **no overlapping tables** — all ten tables verified individually
- ✅ **no overlapping figures** — both embedded images are exactly one column wide, in column 2
- ✅ **all tables readable** — body text 8 pt throughout; no `\resizebox` was used anywhere
- ✅ **all figures readable** — regenerated at 300 dpi with 7–8 pt axis text
- ✅ **author block readable** — five names on one line, correct order, inside margins
- ✅ **references readable** — 13 entries, balanced across two columns on p8
- ✅ **no heading/table or heading/body collisions** — verified on every page

## 5. Known cosmetic items (not defects)

1. Page 1 ends column 2 with the bold run-in heading "Contributions." immediately before
   the list. Acceptable in IEEEtran; can be nudged with a `\pagebreak` if a venue objects.
2. Page 8 remains partly empty below the references. This is normal for a paper ending in
   its bibliography and cannot be removed without padding the content.
3. The paper is 8 pages. **Check this against the target venue's limit** — many IEEE
   conferences cap camera-ready at 6 pages plus references.
