# SQuAD 2.0 Answerability Protocol under Retrieval

**Status:** RESOLVED — decided before any experiment was run
**Date:** 12 August 2026
**Decision ID:** D-002
**Author:** VeriQA implementation
**Binding on:** corpus construction, correctness labels, all splits, E0–E7

---

## 1. The problem

SQuAD 2.0's answerability label is **paragraph-relative**. When a crowdworker marked a question `is_impossible = true`, they meant *"this question cannot be answered from **this specific paragraph**."* They did not mean *"this question cannot be answered from anywhere."*

VeriQA introduces a retrieval stage. The reader no longer sees one designated paragraph — it sees whatever the retriever returns from a corpus. The moment the corpus contains more than the gold paragraph, the original label may no longer describe the situation being evaluated.

Two distinct failures follow if this is ignored:

**Failure 1 — invalid abstention labels.** A question labelled unanswerable may in fact be answerable from some *other* paragraph now in the corpus. The system answers it correctly, and we score that as an error. Every abstention metric is then measuring the wrong thing.

**Failure 2 — invalid answerable labels.** A question labelled answerable may have its gold paragraph outside the retrieval reach, making it unanswerable in practice. We score a correct abstention as a failure.

Failure 2 is straightforward to prevent — keep the gold paragraph in the corpus. **Failure 1 is the dangerous one, because it is silent.** Nothing in the data announces it.

---

## 2. Measurement, not assumption

We measured the size of the problem before choosing a protocol.

For every unanswerable question, SQuAD 2.0 records a `plausible_answer` — the span a crowdworker judged to be the most tempting wrong answer, drawn from the gold paragraph. If that exact string also occurs in another paragraph now inside the retrieval corpus, there is a concrete pathway by which the question could become answerable and the label could become invalid.

We counted this on a random sample of 40 training articles (seed 20260814).

| Retrieval corpus design | Plausible answer also present elsewhere in corpus | |
|---|---|---|
| | **All plausible answers** (n = 4,522) | **Substantive only** (n = 3,049) |
| **A · Article-pooled** — corpus contains all paragraphs of the question's own article | **28.70 %** | **14.89 %** |
| **B · Sibling-masked** — corpus contains the gold paragraph plus paragraphs from *other* articles | 14.75 % | **2.39 %** |

*Substantive* = at least two tokens, not purely numeric, at least six characters. The unfiltered figures are inflated by short coincidental strings — bare years such as "1997" match almost any corpus and carry no evidence that the question became answerable. The substantive column is the one that reflects genuine label risk.

**Result: pooling an article's own paragraphs puts roughly one in seven substantive unanswerable labels at risk. Masking siblings reduces that to about one in forty — a 6.2× reduction.**

A 14.9 % label-validity failure rate would be fatal. Roughly half of the dev set is unanswerable, so corrupting a seventh of those corrupts about 7 % of all evaluation labels — larger than any effect we expect to measure. Protocol A is therefore rejected on measured evidence, not on intuition.

---

## 3. Adopted protocol — B, sibling-masked cross-article distractors

### 3.1 Corpus construction

For each split *S* (an article-disjoint partition, see §4), the corpus is

> **C_S** = every paragraph belonging to every article in *S*

Paragraphs are used whole. SQuAD paragraphs have a median length of 107 words (p90 = 180), which sits inside the reader's window, so **no sub-paragraph chunking is applied.** This is a deliberate departure from the 256-token chunking in the original design document: chunking would split gold answer spans across chunk boundaries and introduce a confound that has nothing to do with the research question. The paragraph is the natural retrieval unit for this dataset.

### 3.2 Retrieval space for a single question

For a question *q* whose gold paragraph is *p*, belonging to article *A*, the retrieval space is

> **R_q = C_S \ { paragraphs of A other than p }**

In words: the question can retrieve **its own gold paragraph** and **any paragraph from any other article**, but never a sibling paragraph from its own article.

### 3.3 Why this makes both labels valid

**Answerable questions.** The gold paragraph is always in *R_q*. A correct answer is therefore always reachable. If the retriever fails to rank it into the top-*k*, that is a genuine retrieval failure of exactly the kind the system is meant to detect — not a labelling artefact.

**Unanswerable questions.** The only same-topic paragraph in *R_q* is the gold paragraph, from which the question is unanswerable by SQuAD's own construction. All other paragraphs come from different articles on different subjects. The measured residual risk is 2.39 %, and inspection shows those are dominated by coincidental short-string matches rather than genuine answerability.

**The masking is applied identically to answerable and unanswerable questions**, and uses only the article identifier — never the answerability label. No label information enters corpus construction. This matters: masking siblings *only* for unanswerable questions would leak the label into the retrieval space and invalidate the entire experiment.

### 3.4 Precedent

This is the *distractor setting*, established practice in multi-document QA — gold evidence placed among distractor passages so that retrieval is genuinely challenged while answerability remains controlled. HotpotQA uses the same construction. We are not inventing an evaluation format; we are applying a known one and documenting why.

---

## 4. Splits

Splits are **by article**, never by question. SQuAD contains a median of 279 questions per article, many concerning the same facts; a question-level split places near-duplicates on both sides and inflates every metric.

A three-way split is not sufficient here, because two different models must be trained. The reader must not be evaluated on articles it trained on, *and* the reliability model must not be trained on reader outputs from articles the reader has already seen — a reader is unrepresentatively accurate on its own training articles, so its error distribution there is not the distribution the reliability model will face at test time.

| Partition | Share | Purpose | Sees gold answers? |
|---|---|---|---|
| `reader_train` | 40 % | Fits the extractive reader only | Yes |
| `rel_train` | 20 % | Fits the reliability classifier on reader outputs | Reader outputs only |
| `rel_calib` | 10 % | Isotonic calibration + risk-threshold selection | Reader outputs only |
| `test` | 30 % | All reported results | Never touched until the final run |

**Invariant, enforced by test:** the four article sets are pairwise disjoint. `tests/test_no_leakage.py` asserts this and fails the build otherwise.

Corpus A and Corpus B (used for the domain-shift experiment, E6) are drawn from `train-v2.0.json` and `dev-v2.0.json` respectively. Their article titles were checked and **overlap is zero**, so E6 is a genuine transfer test.

---

## 5. Correctness label

The original research design defined correctness as SQuAD F1 ≥ 0.5 for answerable questions and a correct no-answer for unanswerable ones. **That definition survives protocol B unchanged**, because protocol B was chosen precisely to keep it valid. No adjustment is required.

Formally, for a question *q* with system output *â*:

- *q* **answerable**: `correct(q) = max over gold answers a of F1(â, a) ≥ 0.5`, and `correct = False` if the system abstained or returned no answer.
- *q* **unanswerable**: `correct(q) = True` if and only if the system returned no answer.
- The reliability target is `error(q) = ¬correct(q)`.

SQuAD's official normalisation is applied before scoring: lowercase, strip articles (*a*, *an*, *the*), strip punctuation, collapse whitespace.

**Sensitivity.** The 0.5 threshold is a choice. The evaluation re-runs the headline comparison at 0.4 and 0.6 and reports whether the ranking of methods changes. If it does, that is reported as a limitation rather than hidden.

---

## 6. Residual limitations — stated, not hidden

1. **2.39 % residual label risk** on substantive unanswerable questions remains. It is measured, reported, and small relative to the effects under study, but it is not zero.
2. The plausible-answer string test is a **proxy**. A string can be present without the question being answerable there, and a question could conceivably be answerable through paraphrase with no string match. The proxy bounds the risk; it does not eliminate it.
3. SQuAD 2.0's unanswerable questions were **written adversarially** to look plausible. They may be systematically easier or harder to detect than unanswerable questions arising naturally in a deployed system. We cannot resolve this with this dataset and do not claim to.
4. Paragraph-level retrieval means results **do not transfer directly to systems that chunk long documents**. A deployment over 50-page PDFs faces a chunking problem this evaluation deliberately avoids.

---

## 7. Decisions recorded

| ID | Decision | Rationale | Supersedes |
|---|---|---|---|
| **D-002** | Adopt sibling-masked cross-article distractor corpus (protocol B) | Measured 6.2× reduction in substantive label contamination, 14.89 % → 2.39 % | Article-pooled corpus implied by the original design |
| **D-003** | No sub-paragraph chunking; the paragraph is the retrieval unit | Median paragraph is 107 words, inside the reader window. Chunking would split gold spans and confound the measurement | 256-token chunking in `01_WINNING_PROJECT_DESIGN.md` |
| **D-004** | Four-way article split, adding `reader_train` | The reader and the reliability model are distinct models; the reliability model must see a realistic reader error distribution | Three-way split in `02_RESEARCH_DESIGN.md` |
| **D-005** | Correctness definition retained unchanged at F1 ≥ 0.5 | Protocol B was selected to preserve its validity; sensitivity checked at 0.4 / 0.6 | — |

All four were fixed **before** any experiment was executed.
