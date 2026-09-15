# Decisions Log — append only, newest at the bottom

Format: `D-nnn · decision · rationale · evidence`

### D-001 · Project selection: VeriQA
Selective extractive QA with a calibrated abstention gate, chosen from 15 scored candidates
(weighted score 8.75 vs 7.91 runner-up). See `../../NEW_PROJECT_SELECTION/`.

### D-002 · Sibling-masked cross-article retrieval corpus
**Decision:** a question may retrieve its own gold paragraph and any paragraph from a *different*
article, never a sibling paragraph of its own article.
**Rationale:** measured. Article-pooled corpora leave 14.89% of substantive unanswerable
questions with their plausible-answer string elsewhere in the corpus; sibling masking cuts this
to 2.39%, a 6.2× reduction. Pooling would have corrupted several percent of all evaluation labels.
**Evidence:** `research/SQUAD_ANSWERABILITY_PROTOCOL.md` §2.

### D-003 · No sub-paragraph chunking
Median SQuAD paragraph is 107 tokens, inside the reader window. Chunking would split gold spans
across boundaries and add a confound orthogonal to the research question. Supersedes the
256-token chunking in the original design.

### D-004 · Four-way article split, adding `reader_train`
The reader and the reliability model are different models. A reliability model fitted on articles
the reader trained on would see an unrepresentatively optimistic error distribution. Split is
40/20/10/30 by article. Enforced by `tests/test_no_leakage.py`.

### D-005 · Correctness definition retained at F1 ≥ 0.5
Protocol B was chosen precisely to keep it valid, so no adjustment was needed. Sensitivity
checked at 0.4 and 0.6; the headline finding holds at all three (`results/tables/T8_*.csv`).

### D-006 · Question tokenisation bug fix
`question.lower().split()` left trailing punctuation attached, so the final and often most
informative content word of every question never matched a passage token. Switched to regex
tokenisation in both the reader and the retrieval feature family. Reader F1≥0.5 on a 25-article
pilot rose from 0.112 to 0.121.

### D-007 · Candidate span filter
Spans beginning or ending on a stopword or punctuation token are discarded. Measured on 25
articles: gold-answer recall 92.69% → 90.35% (−2.3 points) while candidates fall 2.8×. The
recall cost buys a far easier discrimination problem for the span scorer.

### D-008 · Two-stage reader (sentence selection, then span scoring)
Measured on 20 articles: 68.2% of gold answers lie in the single highest-overlap sentence, 78.7%
in the top two, 83.3% in the top three. Restricting candidates to the top three sentences raised
pilot F1≥0.5 from 0.121 to 0.158.

### D-009 · Classical reader backend instead of a transformer
Hugging Face is unreachable from the build environment by every route tested. Substituted
TF-IDF+SVD embeddings and a feature-based span scorer with a trained null head. Documented in
full in `implementation/READER_BACKEND.md`. **This is the most consequential limitation of the
project and is disclosed in the abstract, Section XI and the conclusion of the paper.**

### D-010 · Metric polarity bug caught after the first full run
The first E1 run passed the *error* array into parameters expecting *correct*, inverting every
metric — it reported P1 at ROC-AUC 0.361 (below chance) and B1 at AURC 0.111 against an error
rate of 0.886, both impossible. Caught by inspecting the numbers, not by the tests: the negative
control still passed, because a shuffled label is symmetric under inversion. Fixed by making the
polarity explicit at the call site and adding an assertion in `all_metrics`. **Recorded because
it shows a negative control alone does not catch every class of bug.**
