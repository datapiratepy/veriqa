# VeriQA — Demo and Screenshot Guide

Every question below was **executed against the live service in this session** and the
observed output is quoted. Nothing here is invented. If your output differs, something is
wrong with your setup — tell me.

**Start:** `streamlit run app\streamlit_app.py` → http://localhost:8501

---

## The two demos

### DEMO 1 — SUPPORTED QUESTION → the system answers

Use one of these five. All were verified live: answered (not abstained) **and** correct
against the SQuAD gold answer.

| Question | Answer you should see | Risk | Article |
|---|---|---|---|
| **Where was the band Blue Cheer from?** | `San Francisco` | 0.5714 | Hard_rock |
| What year did the Island's flax mills close? | `1965` | 0.5714 | Saint_Helena |
| What is the color rendering index of an incandescent light? | `100` | 0.5714 | Incandescent_light_bulb |
| Why are dogs viewed as unclean in Islam? | `scavengers` | 0.5714 | Dog |
| What is the fourth and final stomach compartment in ruminants? | `abomasum` | 0.5714 | Digestion |

**Recommended:** *"Where was the band Blue Cheer from?"* — short, unambiguous, and the
answer span highlights cleanly in the passage.

**What you should see:** a green box with the answer; "Estimated probability of error:
**0.57** (threshold 0.57)"; the supporting passage with the span highlighted orange; three
timing metrics; and the two expanders.

### DEMO 2 — UNSUPPORTED QUESTION → the system abstains

All four verified live. All abstained.

| Question | Risk | Suppressed candidate | Retrieved from |
|---|---|---|---|
| **Who won the 2026 FIFA World Cup final?** | 0.9706 | `1998` | Paris |
| What was the budget allocated for the 2027 Bengaluru metro extension? | 0.9556 | `7.6 billion` | Paris |
| What is the annual revenue of Tesla in 2025? | 0.8898 | `980` | Plymouth |
| How many students are enrolled at AMC Engineering College? | 0.9556 | `597` | University_of_Kansas |

**Recommended:** *"Who won the 2026 FIFA World Cup final?"* — **this is the strongest
moment in the whole demo.** The corpus contains no football results, the reader nonetheless
extracts `1998` from a Paris article, and the reliability layer suppresses it.

**What you should see [VERIFIED]:** a red box reading *"I cannot answer this reliably from
the provided documents."*; "Estimated probability the answer would be wrong: **0.97**
(threshold 0.57)"; a **Why:** list containing *"no corroboration in the second passage"* and
*"retrieved passages disagree"*; and an expander revealing the suppressed `1998`.

**Say this out loud during the demo:** *"Without the reliability layer, that 1998 is what
the user would have seen — a confident, specific, completely fabricated answer."*

---

## ⚠️ Two things to know before you demo

**1. The risk slider is stepped, not smooth.** Isotonic calibration emits only nine
distinct values (0.5714, 0.8254, 0.8521, 0.8750, 0.8898, 0.9447, 0.9556, 0.9706, 1.0), and
**only 0.5714 falls below the default threshold**. Dragging the slider between 0.60 and
0.82 changes nothing visible. To show it flipping a decision, take a Demo-1 question and
drag the threshold **below 0.5714** — it will switch from answering to abstaining.

**2. Most questions abstain.** Base accuracy is 0.1144, so if you improvise a question it
will very likely abstain. **Use the verified questions above.** This is not a bug; it is the
measured behaviour reported in the paper.

**3. There is no document upload.** Do not promise the examiner they can upload a PDF —
that control does not exist. The corpus is fixed at 5,637 SQuAD passages.

---

## Screenshots to capture

Only functionality that actually exists is listed.

### SCREENSHOT 1 — Application home screen
**Do:** open http://localhost:8501, leave the default question, do not click Ask.
**Shows:** title, caption ("CPU only: no GPU, no API key, no network"), the three tabs, the
question box, both sliders, and the `Corpus passages 5,637` metric.
**Why:** establishes that a real, running application exists.
**Caption:** *"Fig. X. VeriQA interface. The risk-tolerance slider exposes the
abstention threshold as a user-facing control."*
**Evidence type:** 🖥️ **UI demonstration** — proves the application is implemented and runs.

### SCREENSHOT 2 — Supported question → answer + evidence
**Do:** ask *"Where was the band Blue Cheer from?"*, click Ask, scroll so the answer and the
highlighted passage are both visible.
**Shows:** green answer box, risk vs threshold, supporting passage with the span highlighted.
**Why:** demonstrates the full retrieve → read → decide path.
**Caption:** *"Fig. X. A supported question. The system returns the extracted span with its
source passage and estimated error probability."*
**Evidence type:** 🖥️ **UI + pipeline demonstration.**

### SCREENSHOT 3 — Unsupported question → abstention ⭐
**Do:** ask *"Who won the 2026 FIFA World Cup final?"*, click Ask, **expand "Show the
candidate answer that was suppressed"** before capturing.
**Shows:** red abstention box, risk 0.97 vs threshold 0.57, the two reasons, and the
suppressed `1998`.
**Why:** **this is the single most important screenshot.** It shows both halves of the
central idea at once — the wrong answer that would have been shown, and the decision not to
show it.
**Caption:** *"Fig. X. An unsupported question. The reader still proposes a span (`1998`),
but the reliability layer estimates a 0.97 probability of error and abstains, reporting the
signals responsible."*
**Evidence type:** 🖥️ **UI demonstration of selective-prediction behaviour.** It illustrates
the mechanism; it does **not** quantify it.

### SCREENSHOT 4 — The 16 reliability features
**Do:** on either result, expand *"Reliability features (the 16 signals behind the
decision)"*.
**Shows:** all 16 named features with live values.
**Why:** proves the feature set in the paper is genuinely computed at inference, not a
diagram.
**Caption:** *"Fig. X. The sixteen reliability features computed for a single query,
spanning retrieval, reader and cross-evidence agreement."*
**Evidence type:** 🖥️ **Implementation evidence** — the strongest UI screenshot for showing
the method is real.

### SCREENSHOT 5 — Retrieved evidence
**Do:** expand *"All retrieved evidence"*.
**Shows:** five passages with article names and fused retrieval scores.
**Why:** shows hybrid retrieval returning ranked evidence.
**Caption:** *"Fig. X. Top-k retrieved passages with fused LSA+BM25 scores."*
**Evidence type:** 🖥️ **UI demonstration.**

### SCREENSHOT 6 — Frozen results dashboard
**Do:** click the **Frozen results** tab.
**Shows:** the four headline metrics, the 4-PASS/3-FAIL gate table, the three figures, and
four result tables — all read from `results/RESULT_FREEZE.json`.
**Why:** shows the experimental results are stored as artefacts and surfaced by the app.
**Caption:** *"Fig. X. Frozen experimental results as presented in the application,
including the pre-registered gate outcomes."*
**Evidence type:** 📊 **Pointer to experimental evidence.** The screenshot proves the app
displays the frozen file; the *numbers* are evidenced by the CSVs, not by this image.

### SCREENSHOT 7 — Test suite passing (terminal)
**Do:** run `python -m pytest tests -q` and capture the terminal.
**Shows:** `19 passed`.
**Why:** evidence of engineering discipline, including the leakage test.
**Caption:** *"Fig. X. Test suite, including article-disjointness (leakage) checks and
hand-computed metric fixtures."*
**Evidence type:** ⚙️ **Implementation evidence.**

---

## Demo evidence vs experimental evidence

**This distinction matters and examiners probe it.** A screenshot never proves a number.

| Evidence | What it actually proves | What it does **not** prove |
|---|---|---|
| Running UI (S1) | The application is implemented and runs on a laptop | Nothing about accuracy or AUC |
| Answer + retrieved passage (S2) | The retrieve→read→decide pipeline works end to end | That the answer is *usually* right |
| Abstention output (S3) | Selective-prediction behaviour is implemented and fires | That abstention is *well calibrated* |
| 16-feature panel (S4) | The features in the paper are genuinely computed at inference | That they carry predictive value |
| Retrieved evidence (S5) | Hybrid retrieval returns ranked passages | That retrieval is competitive |
| Terminal `19 passed` (S7) | Tests exist and pass, including leakage checks | That the results generalise |
| **`results/tables/T2_main_comparison.csv`** | **P1 ROC-AUC 0.6390 vs B5 0.5242** | — |
| **`results/raw/E2_p1_vs_b5.json`** | **ΔAURC −0.0345, interval [−0.0495, −0.0191]** | — |
| **`results/tables/T4_ablation.csv`** | **Which feature family carries the effect** | — |
| **`results/RESULT_FREEZE.json`** | **All seven gate outcomes, 4 PASS / 3 FAIL** | — |

**The rule:** UI screenshots are evidence of *implementation*. The CSV and JSON artefacts
are evidence of *results*. If an examiner asks "how do you know retrieval features help?",
the answer is Table IV and the paired bootstrap — **not** the screenshot.

---

## If something goes wrong on demo day

| Symptom | Fix |
|---|---|
| App slow on first question | Normal — 57 MB model loads once. Ask a warm-up question before starting. |
| Your improvised question abstains | Expected. Return to the verified list. |
| Slider seems to do nothing | Expected — nine discrete risk values. Drag below 0.5714 on a Demo-1 question. |
| Answer not highlighted in the passage | Highlighting is literal string matching; it fails when the span differs in case or whitespace. Cosmetic only. |
| `ModuleNotFoundError: veriqa` | Run from the `VeriQA` folder with the venv active. |
| Port 8501 busy | `streamlit run app\streamlit_app.py --server.port 8502` |
