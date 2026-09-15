# Viva Guide

Answers below are grounded in the frozen results. Numbers are quoted from
`results/RESULT_FREEZE.json` and `results/tables/`.

## Basic

**What is VeriQA?** A document question-answering system that estimates the probability its own
answer is wrong and abstains when that risk is too high. Retrieval, extractive reading, then a
calibrated reliability layer that decides whether to speak.

**What problem does it solve?** Retrieval-augmented systems answer everything, including
questions the documents cannot answer, and a fabricated answer looks exactly like a correct one.

**Why abstain instead of improving accuracy?** Different problem. Even a perfect reader cannot
answer a question whose answer is absent — 29.5% of our test questions are like that. Abstention
is the only correct response there.

**What is the dataset?** SQuAD 2.0. Corpus A is 120 training articles — 5,637 passages, 11,921
questions. Corpus B is the full dev set, used only to test transfer. Article titles are disjoint.

**Why CPU-only?** The users most exposed to this problem — departments, small offices — have no
GPU budget. It was also forced on us: no pretrained weights were reachable from our build
environment.

## Technical

**Why extractive rather than generative?** It runs in 20 ms on CPU, returns a span you can point
at in a real passage, and exposes a null-answer score — our strongest single-feature baseline.

**Why hybrid retrieval?** LSA captures paraphrase; BM25 recovers exact names and numbers that a
256-dimensional projection blurs. It also gives us a lexical-overlap feature, which turned out to
be the single most load-bearing feature in the whole model (permutation ΔAURC 0.0121).

**Why these 16 features?** They cover the three places information about correctness lives. The
set is small deliberately: with 2,400 training rows a larger set overfits and permutation
importance stops being interpretable.

**Why gradient boosting?** Small tabular feature set, non-linear interactions likely, trains in
seconds, supports permutation importance. A neural calibrator on 2,400 rows would overfit and
explain nothing.

**Why isotonic calibration separately?** The classifier ranks well but its raw output is not a
probability. Isotonic on a held-out split maps score to observed error frequency — ECE falls from
0.0895 to 0.0120. Without it we could rank but not set an operating point.

**Why AURC and not accuracy?** Accuracy is one operating point; a selective system has a whole
curve. AURC summarises it across all thresholds. We also report selective accuracy at fixed
coverage because that is what a deployer reads.

**Why article-level splits?** SQuAD has a median of 279 questions per article, many about the
same facts. A question-level split puts near-duplicates on both sides and inflates everything.
We use four partitions, not three, because the reader and the reliability model are different
models — a reliability model trained on articles the reader memorised would see an
unrepresentatively optimistic error distribution.

**Why cluster bootstrap?** Questions within an article are correlated. Resampling questions gives
intervals that are far too narrow. We resample articles.

**How do you prevent leakage?** Splits are created in exactly one function; `tests/test_no_leakage.py`
asserts the four article sets are pairwise disjoint and that no question appears in two
partitions. And we run a negative control: shuffle the error labels, refit, and confirm ROC-AUC
returns to chance. Ours gave 0.4887 against a pre-registered band of 0.45–0.55.

## Research

**What is actually your contribution?** The measurement, not the mechanism. Trained calibrators
for QA abstention are Kamath, Jia and Liang, ACL 2020 — we reimplement their approach as our own
baseline B5 and credit it as theirs. What had not been measured is whether retrieval-side signals
add anything in a CPU-only extractive pipeline. They do: ROC-AUC 0.524 → 0.639, ΔAURC −0.0345
with a bootstrap interval excluding zero.

**What is inherited from Kamath et al.?** The idea of training a separate classifier to predict
reader error rather than thresholding softmax. Their setting had no retrieval stage; that is the
gap we work in.

**Why would retrieval information help?** The reader's confidence is conditioned on a passage it
was handed — it says which span is best, not whether the passage should have been shown. 29.5% of
our test questions have no answer in the corpus and another 6.1% never retrieve their evidence;
over a third of errors are therefore invisible in principle to a reader-side signal.

**What if the proposed method had lost?** We wrote the negative-result paper into the plan before
running anything. It didn't lose, but three of seven gates failed and we report those.

**Which gates failed?** G1: base reader accuracy 0.114 against a 0.45 criterion. G2: no method
reached AUC 0.70, best was 0.639. G5: the 80% accuracy target is unreachable — every method,
including ours, sits at zero coverage for it.

**What are the limitations?** Chiefly the reader. It is a classical feature-based span scorer
because Hugging Face was unreachable, so mean answer F1 is 0.145. Every result is conditional on
that. A transformer's null head is trained on 130,000 questions; ours on 4,736, and if reader
features become dominant with a strong reader our conclusion narrows to weak-reader regimes.
Also: 2.39% residual label risk, adversarially-authored unanswerable questions, correlated
features bounding the ablation from below, and a transfer test that moves both corpus and reader.

**What would you improve with more time?** Re-run with a transformer reader — it changes every
absolute number and might change the comparative conclusion, which is exactly why it matters. The
code is backend-agnostic to allow it.

**Isn't this just thresholding a confidence score?** That is baseline B2, and we report it: ROC-AUC
0.4920, indistinguishable from chance. The difference is where the information comes from.

## Who defends what
- **Harsh Ranjan** — ingestion, chunking decision, LSA + BM25, sibling masking, why paragraphs are
  not chunked.
- **Harsh Kamat** — the reader's two stages, the 16 features, gradient boosting, isotonic, the gate.
- **Manish Kumar** — metrics and why each was chosen, cluster bootstrap, the negative control,
  leakage tests, the app.

## Three rules
Never invent a number — "it's in Table II" is a fine answer. When you don't know, say so and say
how you'd find out. Volunteer the limitations before you're asked.
