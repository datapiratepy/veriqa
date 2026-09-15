# Quantifying the Incremental Value of Retrieval-Side Reliability Signals for Selective Extractive Question Answering

**Harsh Kamat**, **Harsh Ranjan**, **Manish Kumar**, **Divya Tyagi**, **Snigdha Kesh**

Department of Computer Science and Engineering
AMC Engineering College, Visvesvaraya Technological University
Bengaluru, India

*Author note: Harsh Kamat, Harsh Ranjan and Manish Kumar carried out the implementation and experiments. Divya Tyagi supervised the work as project guide. Snigdha Kesh served as project coordinator. Institutional e-mail addresses and ORCID identifiers are omitted pending confirmation.*

---

## Abstract

Selective question answering — answering only when the answer is likely correct, and abstaining otherwise — is well established, as are calibrated confidence estimation, retrieval-quality scoring and evidence-sufficiency testing. What is less well quantified is how much *incremental* error-prediction information retrieval-side and cross-evidence signals carry once a reader-side calibrator is already in place, particularly in lightweight pipelines that cannot afford a second large-model inference pass. This paper reports a controlled measurement of exactly that quantity. We implement a complete CPU-only extractive question-answering system — hybrid latent-semantic and BM25 retrieval, a two-stage feature-based span reader with a trained null-answer head, sixteen reliability features spanning retrieval, reader and cross-evidence agreement, gradient-boosted risk estimation and isotonic calibration — and evaluate it on 3,585 SQuAD 2.0 questions drawn from 36 held-out articles. Splits are article-level; a negative control with shuffled labels returns an error-prediction ROC-AUC of 0.4887 against a pre-registered band of 0.45–0.55. The fused reliability representation attains an error-prediction ROC-AUC of 0.6390 (95% bootstrap interval 0.6127–0.6668) compared with 0.5242 for a reader-only calibrator following Kamath *et al.*, and improves the area under the risk–coverage curve by 0.0345 (95% paired bootstrap interval 0.0191–0.0495, excluding zero); the direction and interval hold at correctness thresholds of 0.4, 0.5 and 0.6. Leave-one-family-out ablation attributes the improvement chiefly to retrieval features (ΔAURC 0.0274, interval 0.0149–0.0411) and secondarily to agreement features (0.0103, interval 0.0043–0.0173), whereas removing all five reader features yields an interval containing zero (0.0067, −0.0044–0.0189). The reliability layer adds 1.96 ms to a 63.8 ms median query. These results are conditional on a deliberately weak reader: because pretrained transformer weights were unavailable in our build environment, the reader is a classical feature-based span scorer with a base accuracy of 0.1144, and three of seven pre-registered gates consequently failed, including a target-accuracy gate that no method achieved. We claim the measurement, not the mechanism.

**Index Terms** — selective prediction, question answering, abstention, confidence calibration, uncertainty estimation, information retrieval, resource-constrained inference.

---

## I. Introduction

A document question-answering system that always answers is easy to build and difficult to trust. When the retrieved passages do not contain the answer, the system does not fail visibly: it returns a fluent, well-formed span attached to a real passage, and that output is indistinguishable in form from a correct one. The practical consequence is that every answer must be independently verified, which removes the time saving that motivated the system.

The problem is therefore not only answer extraction but *answer triage* — deciding whether an extracted answer deserves to be shown. Thresholding the reader's own confidence is the obvious first move and a weak one. A reader's span probability is computed conditional on a passage the reader was handed; it expresses which span in that passage is best, not whether the passage should have been retrieved at all. Two failure modes are invisible to it by construction: the corpus does not contain the answer, and the retriever did not surface the passage that does. In our evaluation these account for 29.5% and 6.1% of test questions respectively.

Retrieval-side statistics see precisely that region. They are also free: every retriever already computes similarity scores, rank margins and score distributions, so consuming them costs no additional model pass. The same is true of cross-evidence agreement — whether independently retrieved passages yield the same answer string.

This observation is not new. Selective prediction has a mature theoretical basis [6], [7]; trained calibrators for question answering were established by Kamath, Jia and Liang [1]; large-scale reading calibration was studied by Dhuliawala *et al.* [3]; retrieval-conditioned answerability and evidence sufficiency have been examined in open-domain and retrieval-augmented settings [4], [5]. **We do not claim to have invented retrieval-aware abstention.**

What remains under-quantified is the *increment*. Given a reader-side calibrator of the kind [1] established, how much additional error-prediction information do retrieval-side and cross-evidence signals contribute, measured under controlled conditions with matched splits and interval estimates? And does that increment survive in a pipeline constrained to run on a CPU without pretrained neural weights — the regime in which the cheapness of these signals matters most? This paper answers those two questions empirically.

**Contributions.**

1. A controlled experimental framework for isolating the incremental reliability information contributed by retrieval-side, reader-side and cross-evidence signals, using article-level splits, a negative control and article-clustered bootstrap intervals.
2. A sixteen-feature reliability representation spanning three signal families, computed entirely from artefacts the pipeline has already produced and therefore requiring no additional inference pass.
3. A systematic comparison against always-answer, two single-signal, one reader-native and one reader-only-calibrator baseline, the last being a reimplementation of [1] in our setting.
4. Leave-one-family-out ablation with a permutation-importance cross-check, a domain-transfer evaluation on a disjoint corpus, an error taxonomy, and per-stage latency and memory measurement.
5. The empirical finding that the fused representation improves error discrimination and selective risk relative to the reader-only calibrator in the evaluated CPU-only setting, together with a full account of the pre-registered gates it failed.

## II. Related Work

**Selective prediction.** The formal framework of a predictor paired with a gating function that may decline is due to El-Yaniv and Wiener [6], who established the theory of noise-free selective classification, and was extended to deep networks by Geifman and El-Yaniv [7]. The standard evaluation instrument is the risk–coverage curve; we use its scalar summary, the area under that curve (AURC), as our primary metric.

**Calibration.** Guo *et al.* [8] documented systematic miscalibration in modern neural classifiers and evaluated post-hoc remedies. Isotonic regression as a calibration map is due to Zadrozny and Elkan [9]. We apply the latter and report expected calibration error before and after.

**Calibration and abstention in question answering.** Kamath, Jia and Liang [1] showed that a separate classifier trained to predict whether a QA model erred outperforms thresholding the model's softmax probability, especially under domain shift. Their setting supplies the reader with a passage; there is no retrieval stage, so retrieval-side signals are outside their scope. Our baseline B5 is their approach reimplemented over our reader's features, and we treat it as the reference method to beat rather than as a weak comparator. Dhuliawala *et al.* [3] studied calibration of machine reading at scale, confirming that reading confidence alone is an imperfect error signal. SQuAD 2.0 [2] supplies the unanswerable questions that make abstention measurable at all.

**Answerability and evidence sufficiency in retrieval settings.** The question of whether retrieved context supports an answer has been approached from several directions. Joren *et al.* [4] introduce *sufficient context* as an explicit lens on retrieval-augmented systems, showing that models frequently answer when the retrieved context is insufficient and proposing selective generation guided by a sufficiency signal. Work on answerability in retrieval-augmented open-domain QA [5] shows that models trained to reject randomly paired irrelevant passages generalise poorly to irrelevant passages with high semantic overlap. Lewis *et al.* [10], in the course of introducing PAQ and the RePAQ retriever, demonstrate that retrieval-side scores themselves carry usable confidence information and support selective answering with a tunable accuracy–coverage trade-off.

**Positioning.** Taken together, this literature establishes that (i) selective prediction is well founded, (ii) trained calibrators beat raw reader confidence, and (iii) retrieval context sufficiency is informative. What it does not directly supply is a controlled measurement of the *marginal* contribution of retrieval-side and cross-evidence features **given** a reader-side calibrator, in a pipeline where no second large-model pass is affordable. Several of the closest works [4], [10] operate with large language models or large-scale neural retrievers; the incremental question in a CPU-only extractive regime is what we measure. We claim the measurement, not the mechanism.

## III. Research Question and Scope

**RQ.** Do retrieval-side and cross-evidence reliability signals provide incremental error-prediction value beyond a reader-only calibrator in a CPU-only extractive question-answering pipeline?

Subsidiary questions: whether reader error is predictable at all in this regime; whether a fused representation improves on the reader's own trained null-answer mechanism; whether the resulting risk estimate is calibrated; which feature family carries the contribution; whether the model transfers to a corpus it was not fitted on; and what the layer costs in latency and memory.

**Formulation.** Let *q* be a question, *C* a corpus of passages, *R ⊆ C* the retrieved set and *â* the extracted span. We seek *g(q, R, â) → [0,1]* estimating P(*â* is wrong), together with a threshold *τ* such that answering only when *g < τ* yields a chosen operating point.

**Correctness.** For an answerable question the system is correct if it returns a span with SQuAD F1 ≥ 0.5 against any gold answer; for an unanswerable question it is correct only if it returns nothing. The reliability target is the complement. SQuAD's official normalisation is applied before scoring. The framing presumes a wrong answer is costlier than silence; that is an assumption about deployment, stated rather than proven.

## IV. Dataset and Experimental Protocol

### A. The answerability problem under retrieval

SQuAD 2.0's answerability label is *paragraph-relative*: a question marked unanswerable was judged unanswerable from **its own paragraph**. Introducing retrieval breaks that correspondence. If the corpus contains other paragraphs, an "unanswerable" question may become answerable and the label silently stops describing the situation being evaluated.

We measured the exposure before choosing a design. Every unanswerable SQuAD 2.0 question carries a `plausible_answer`, the most tempting wrong span from its gold paragraph. If that string also occurs elsewhere in the retrieval corpus, there is a concrete route to label invalidation. On a random sample of 40 training articles we counted such occurrences, restricting attention to *substantive* plausible answers (≥ 2 tokens, not purely numeric, ≥ 6 characters), because bare years and other short strings match almost any corpus coincidentally.

**TABLE I: MEASURED LABEL-CONTAMINATION EXPOSURE (n = 3,049 SUBSTANTIVE UNANSWERABLE QUESTIONS)**

| Corpus design | Plausible answer present elsewhere in corpus |
|---|---|
| Article-pooled (sibling paragraphs included) | 14.89% |
| Sibling-masked (gold paragraph + other articles) | **2.39%** |

Pooling an article's own paragraphs would place roughly one in seven substantive unanswerable labels at risk — larger than any effect we intend to measure. We therefore adopt the **sibling-masked** design: the retrieval space for a question comprises its own gold paragraph plus every paragraph belonging to a *different* article, and never a sibling paragraph from its own article.

Answerable labels remain valid because the gold paragraph is always reachable; failure to rank it into the top-*k* is a genuine retrieval failure of the kind under study. Unanswerable labels remain valid because the only same-topic paragraph in scope is the one from which the question was authored to be unanswerable. Masking uses the article identifier alone and is applied identically regardless of label; masking by label would leak the target into the corpus. This is the *distractor* construction familiar from multi-document QA, applied here for label hygiene rather than for difficulty.

Passages are used whole. SQuAD paragraphs have a median length of 107 tokens, within the reader's window, so no sub-paragraph chunking is applied; chunking would split gold spans across boundaries and introduce a confound orthogonal to the research question.

### B. Corpora and splits

Corpus A comprises 120 articles sampled from SQuAD 2.0 train — 5,637 passages, 11,921 questions, 31.17% unanswerable. Corpus B, used only for transfer, is the full SQuAD 2.0 dev set — 35 articles, 1,204 passages, 3,500 questions, 50.46% unanswerable. Article titles are disjoint between the two.

Splits are **by article**, never by question: SQuAD carries a median of 279 questions per article, many concerning the same facts, and a question-level split places near-duplicates on both sides. Three partitions are insufficient here because two distinct models are trained. A reliability model fitted on articles the reader has already memorised would observe an unrepresentatively optimistic error distribution. Corpus A is therefore partitioned 40/20/10/30 by article into `reader_train` (48 articles), `rel_train` (24 articles, 2,400 questions), `rel_calib` (12 articles, 1,200 questions) and `test` (36 articles, 3,585 questions). Pairwise disjointness is asserted by a unit test that fails the build.

### C. Pre-registration and statistics

Seven gates with numeric criteria were fixed before any experiment was executed and none was subsequently altered. All runs use seed 20260814; hyperparameters were left at library defaults and were not tuned on test. Confidence intervals derive from a cluster bootstrap over **articles** with 1,000 resamples, because questions within an article are correlated and resampling questions would yield intervals that are far too narrow. Comparisons between two methods use a paired bootstrap on identical article resamples.

## V. System Architecture

The pipeline is conventional up to the point of answering; the contribution occupies one layer.

```
question
   │
   ├─► HYBRID RETRIEVAL
   │     TF-IDF ─► truncated SVD (256) ─► exact inner product
   │     BM25Okapi
   │     min–max normalise, fuse α = 0.5, sibling mask
   │
   ├─► TWO-STAGE EXTRACTIVE READER
   │     1. rank sentences by question overlap, keep top 3
   │     2. enumerate n-grams ≤ 8, drop stopword/punctuation edges
   │     3. gradient-boosted span scorer (20 span features)
   │     4. gradient-boosted null head ─► P(no answer in passage)
   │
   ├─► RELIABILITY LAYER  ★ the measured component
   │     retrieval (7) + reader (5) + agreement (4) = 16 features
   │     gradient boosting ─► isotonic calibration ─► P(error)
   │     threshold selected on calibration split only
   │
   └─► risk > τ ?  yes ─► abstain, with reason
                    no ─► answer + passage + highlighted span + risk
```

Every element shown was implemented and exercised in the reported experiments. No component of this diagram is aspirational.

## VI. Retrieval and Extractive Reader

**Retrieval.** Passages are represented as TF-IDF vectors reduced by truncated SVD to 256 dimensions — latent semantic analysis in the sense of Deerwester *et al.* [11] — and searched by exact inner product. A BM25Okapi index [12] runs in parallel. The two score vectors are min–max normalised over the candidate set and fused with equal weight. The dense projection captures paraphrase; the lexical index recovers exact names, numbers and codes that a 256-dimensional projection blurs, and supplies the lexical-overlap statistic that later proves the single most load-bearing reliability feature.

**Reader.** Stage one ranks the passage's sentences by question-content-word overlap and retains the best three. Measured on training articles, 68.2% of gold answers lie in the highest-overlap sentence and 83.3% within the top three, so this discards most of the candidate space at modest recall cost. Stage two enumerates token n-grams up to length eight within the retained sentences, discards any span beginning or ending on a stopword or punctuation token — a filter retaining 90.4% of gold answers while removing 64% of candidates — and scores the remainder with a gradient-boosted classifier over twenty span features covering capitalisation, numeracy, inverse document frequency, distance to the nearest question term, sentence overlap, left and right context overlap and question-type interactions. A separately fitted gradient-boosted **null head**, trained on the same articles against the `is_impossible` label, emits P(this passage contains no answer). The reader therefore exposes the same interface as a transformer SQuAD 2.0 reader — a best span, a span score and an explicit no-answer score — without pretrained weights.

**Implementation constraint, stated plainly.** The original design specified a DistilBERT SQuAD 2.0 reader and MiniLM sentence embeddings. Neither was obtainable: every route to the model repository was blocked in our build environment. Rather than fabricate results or abandon the experiment, we substituted established methods that require no pretrained weights and preserved the interface on which the research question depends. **No transformer, neural embedding or pretrained language model is used anywhere in the reported system.** The consequences for absolute performance are quantified in Sections IX and XII.

## VII. Reliability Features and Calibration

Sixteen features in three families, all read from artefacts the pipeline has already produced.

**TABLE II: THE SIXTEEN RELIABILITY FEATURES**

| Family | Features |
|---|---|
| Retrieval (7) | top-1 fused score; margin between top two; mean over top-*k*; entropy of the score distribution over a wider candidate list; count above a similarity floor; question–passage lexical overlap; question length |
| Reader (5) | span softmax probability; null-head score; their difference; span length; entropy of the span score distribution |
| Agreement (4) | number of top-*k* passages yielding a non-null answer; edit similarity between answers from the first two passages; fraction of top-*k* answers matching the modal answer; whether the chosen answer string also appears in the second passage |

A gradient-boosted classifier maps this vector to a risk score. Isotonic regression [9], fitted on the held-out calibration split, maps that score to an estimated error probability. A threshold is then selected on the same calibration split.

Two clarifications matter. First, **isotonic calibration adjusts the mapping from score to observed error frequency; it confers no guarantee.** The system estimates risk and applies a threshold; it does not certify that any accuracy level will be attained, and Section IX-D reports a case where the intended operating point was not reached. Second, no feature triggers an additional model pass. That constraint is what holds the layer's cost at 1.96 ms rather than doubling inference time, and it is why several richer signals available in the literature were deliberately excluded.

## VIII. Baselines and Ablation Design

**TABLE III: METHODS UNDER COMPARISON**

| ID | Method | Signal |
|---|---|---|
| B1 | Always answer | none (floor) |
| B2 | Reader span-probability threshold | reader softmax |
| B3 | Retrieval similarity threshold | top-1 fused score |
| B4 | Reader null-answer score | the reader's own trained no-answer head |
| B5 | Reader-only calibrator, after [1] | the 5 reader features |
| P1 | Fused calibrator (evaluated) | all 16 features |

All six consume identical pipeline outputs on identical splits; only the abstain/answer signal differs. **The comparison that answers the research question is P1 against B5**, not P1 against B1, and we report it as such.

The ablation is leave-one-family-out plus each family alone — seven configurations fixed in advance. Permutation importance on the full model provides an independent attribution. A negative control, in which error labels are shuffled before refitting, is run first; if it does not return to chance, no downstream number is reported.

## IX. Experimental Results

### A. Negative control

Shuffling the error labels and refitting yields an error-prediction ROC-AUC of **0.4887** on test, inside the pre-registered band [0.45, 0.55]. Gate G0 passes. No leakage is detectable through the features, the splits or the calibration.

### B. Main comparison

**TABLE IV: SELECTIVE PREDICTION ON 3,585 TEST QUESTIONS FROM 36 HELD-OUT ARTICLES. INTERVALS ARE 95% CLUSTER BOOTSTRAP OVER ARTICLES. AURC: LOWER IS BETTER.**

| Method | Error ROC-AUC | 95% interval | AURC | 95% interval | SelAcc@0.5 | SelAcc@0.7 |
|---|---|---|---|---|---|---|
| B1 always answer | 0.5000 | — | 0.8890 | 0.8455–0.9162 | 0.0988 | 0.1064 |
| B2 span probability | 0.4920 | 0.4544–0.5304 | 0.8823 | 0.8498–0.9084 | 0.1066 | 0.1143 |
| B3 retrieval similarity | 0.5634 | 0.5353–0.5925 | 0.8686 | 0.8362–0.8962 | 0.1328 | 0.1263 |
| B4 reader null head | 0.5526 | 0.5216–0.5823 | 0.8683 | 0.8365–0.8982 | 0.1267 | 0.1235 |
| B5 reader-only calibrator [1] | 0.5242 | 0.5080–0.5428 | 0.8785 | 0.8412–0.9019 | 0.1071 | 0.1112 |
| **P1 fused (evaluated)** | **0.6390** | **0.6127–0.6668** | **0.8423** | **0.8029–0.8731** | **0.1462** | **0.1355** |

Two observations from Table IV are worth isolating. B2 — thresholding reader confidence, the common practitioner default — returns an interval spanning 0.5 and is therefore not distinguishable from chance in this setting. And both single-signal retrieval-aware baselines, B3 (0.5634) and B4 (0.5526), exceed the reader-only *calibrator* B5 (0.5242), despite B5 having five features and a fitted model behind it.

### C. The incremental measurement

Paired bootstrap over articles, identical resamples for both methods:

- **ΔAURC (P1 − B5) = −0.0345, 95% interval [−0.0495, −0.0191]** — the interval excludes zero. Gate G3 passes.
- ΔAURC (P1 − B4) = −0.0293, 95% interval [−0.0442, −0.0150] — excludes zero. Gate G4 passes.

We report these as bootstrap intervals excluding zero. We have not performed a null-hypothesis significance test and do not describe the result as statistically significant.

Because the 0.5 correctness threshold is a choice, the comparison was repeated at 0.4 and 0.6.

**TABLE V: SENSITIVITY TO THE CORRECTNESS THRESHOLD**

| F1 threshold | Base error rate | P1 AUC | B5 AUC | ΔAURC | 95% interval | Excludes zero |
|---|---|---|---|---|---|---|
| 0.4 | 0.8798 | 0.6431 | 0.5132 | −0.0438 | −0.0633 to −0.0263 | yes |
| **0.5** | **0.8856** | **0.6390** | **0.5242** | **−0.0344** | **−0.0495 to −0.0200** | **yes** |
| 0.6 | 0.9021 | 0.6352 | 0.5354 | −0.0294 | −0.0445 to −0.0169 | yes |

The direction and the interval property hold at all three thresholds.

### D. Calibration and a failed operating point

Isotonic calibration reduces expected calibration error from **0.0895 to 0.0120**, below the 0.10 criterion.

The intended operating point was not attained. A threshold selected on the calibration split for a target accuracy of 80% achieved 60% accuracy there at 0.42% coverage, and on test yielded **28% accuracy at 0.7% coverage** — a shortfall of 0.52 against target. **Gate G5 fails.** The cause is visible in Table IV: base accuracy is 0.1144, and although the risk model ranks better than chance it is not discriminative enough to isolate an 80%-accurate subset. Every method in Table IV, including P1, reports zero coverage at that target. In this regime the target is not merely missed but unreachable, and **the system must not be described as delivering any guaranteed accuracy level.**

### E. Feature-family attribution

**TABLE VI: LEAVE-ONE-FAMILY-OUT ABLATION. ΔAURC RELATIVE TO FULL; POSITIVE MEANS REMOVAL DEGRADED PERFORMANCE.**

| Configuration | Features | Error AUC | AURC | ΔAURC vs Full | 95% interval | Excludes zero |
|---|---|---|---|---|---|---|
| Full (R+D+A) | 16 | 0.6390 | 0.8423 | — | — | — |
| − Retrieval | 9 | 0.5410 | 0.8698 | **+0.0274** | 0.0149–0.0411 | yes |
| − Agreement | 12 | 0.6021 | 0.8513 | **+0.0103** | 0.0043–0.0173 | yes |
| − Reader | 11 | 0.6202 | 0.8467 | +0.0067 | −0.0044–0.0189 | **no** |
| Retrieval only | 7 | 0.5662 | 0.8640 | +0.0222 | 0.0088–0.0352 | yes |
| Agreement only | 4 | 0.5680 | 0.8694 | +0.0283 | 0.0138–0.0420 | yes |
| Reader only | 5 | 0.5242 | 0.8785 | +0.0345 | 0.0191–0.0495 | yes |

Removing the retrieval family costs roughly four times what removing the reader family costs, and removing all five reader features produces an interval containing zero. Permutation importance on the full model agrees in direction: the most load-bearing features are `r_lex_overlap` (retrieval, ΔAURC 0.0121 when shuffled), `a_answer_in_top2` (agreement, 0.0107), `d_span_len` (reader, 0.0103), `d_span_prob` (reader, 0.0078) and `r_sim_top1` (retrieval, 0.0066). The reader's dedicated no-answer feature `d_null_score` scores −0.0017, indistinguishable from noise.

One caveat attaches to any ablation over correlated features: because the families overlap in what they encode, leave-one-out yields a **lower bound** on a family's contribution rather than an exact attribution. Agreement between the two attribution methods strengthens the reading without making it exact.

## X. Error Analysis and Domain Transfer

**TABLE VII: OUTCOME TAXONOMY OVER ALL 3,585 TEST QUESTIONS**

| Category | Count | Fraction |
|---|---|---|
| Wrong span (gold passage retrieved, no overlap) | 1,844 | 51.4% |
| Unanswerable question answered anyway | 1,056 | 29.5% |
| Correct | 410 | 11.4% |
| Retrieval failure (gold passage not in top-*k*) | 219 | 6.1% |
| Partial overlap below threshold | 56 | 1.6% |

The dominant failure is span selection rather than retrieval. The gold passage reaches the top-5 for 89.1% of questions and the reader then selects a non-overlapping span in over half of all cases; only 6.1% fail because retrieval missed. This is the quantitative form of the limitation in Section XII: **the reader, not the retriever, is the bottleneck.**

Accuracy varies with question type. Types with a constrained answer form fare best — *when* 18.8% (n = 256), *how many* 18.7% (n = 257), *who* 16.1% (n = 366) — because the span features encode numeracy and capitalisation. Open-ended types fare worst: *what* 8.9% (n = 1,713), *why* 1.7% (n = 59). A reader with no syntactic or semantic representation has little to exploit on a *why* question.

Three verbatim cases from the test set illustrate the behaviour. For "How many passengers came through Brasília's airport in 2007?" the reader returned **11,119** against a gold answer of **11,119,872** — a truncated numeric span scoring below threshold — and the risk model assigned it the maximum risk in the test set, correctly. For "What is the fourth and final stomach compartment in ruminants?" it returned **abomasum** against gold **The abomasum**, correct after normalisation and assigned the lowest risk. For the unanswerable "Until 1988 who was the governor of the Federal District appointed by?" it returned **Brazilian Federal Senate** at low risk: a plausible entity of the expected type located in a high-overlap sentence defeats every feature we compute, and this is precisely the failure the agreement family was intended to catch and does not.

**TABLE VIII: RELIABILITY MODEL FITTED ON CORPUS A, EVALUATED ON CORPUS B (DISJOINT ARTICLES)**

| Method | Setting | n | Error AUC | AURC | ECE |
|---|---|---|---|---|---|
| P1 | A → A in-domain | 3,585 | 0.6390 | 0.8423 | 0.0120 |
| **P1** | **A → B transfer** | **1,000** | **0.6494** | 0.8612 | 0.0143 |
| P1 | B → B in-domain | 1,000 | 0.5680 | 0.8843 | 0.0479 |
| B5 | A → A in-domain | 3,585 | 0.5242 | 0.8785 | 0.0134 |
| B5 | A → B transfer | 1,000 | 0.5177 | 0.9003 | 0.0029 |
| B5 | B → B in-domain | 1,000 | 0.5862 | 0.8773 | 0.0527 |

Under transfer the P1 point estimate shows no large degradation (0.6494 against 0.6390 in-domain); we did not run a paired test across corpora and therefore do not claim the absence of degradation, only that the point estimate does not exhibit it. B5 does not transfer comparably (0.5177). P1 fitted on A also exceeds P1 fitted on B itself (0.6494 against 0.5680), which we read as a training-size effect — Corpus A supplies 2,400 reliability-training questions against B's 700 — rather than as evidence of any advantage from transfer. This setting moves both the corpus and the reader instance simultaneously; it reflects deployment but does not separate the two shifts.

## XI. Computational Cost and Deployment Constraints

**TABLE IX: PER-QUERY LATENCY, SINGLE CPU, n = 7,185 QUERIES (GATE MEASURED OVER 100 REPEATS)**

| Stage | Median (ms) | p95 (ms) |
|---|---|---|
| Retrieval | 43.54 | 60.99 |
| Reader | 19.83 | 23.80 |
| Reliability features | 0.22 | 0.32 |
| Risk model + gate | 1.74 | 2.06 |
| **End-to-end query** | **63.77** | **83.03** |
| **Reliability layer total** | **1.96** | **2.37** |

The reliability layer consumes 1.96 ms, about 3.1% of the query; gate G6 passes with two orders of magnitude of headroom. Retrieval dominates. Peak resident memory was 210.3 MB; index construction over 5,637 passages took 4.64 s and reader fitting 29.2 s. The complete artefact set is 58.4 MB. No network call occurs at query time.

These figures characterise **this** system. They are low principally because no transformer is present, and they should not be read as a claim about a transformer-based pipeline.

**TABLE X: PRE-REGISTERED GATES**

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G0 | negative control ∈ [0.45, 0.55] | 0.4887 | PASS |
| G1 | base reader accuracy ≥ 0.45 | 0.1144 | **FAIL** |
| G2 | some method reaches AUC ≥ 0.70 | 0.6390 | **FAIL** |
| G3 | P1 AURC < B5, interval excludes zero | −0.0345 [−0.0495, −0.0191] | PASS |
| G4 | P1 AURC < B4, interval excludes zero | −0.0293 [−0.0442, −0.0150] | PASS |
| G5 | ECE < 0.10 and target within ±3 points | 0.0120 / shortfall 0.52 | **FAIL** |
| G6 | reliability layer < 100 ms | 1.96 ms | PASS |

Four pass, three fail. The incremental-value question is answered; the absolute-performance criteria are not met.

## XII. Limitations and Threats to Validity

**The reader bounds everything.** Because pretrained transformer weights were unavailable, the reader is a classical feature-based span scorer. Base accuracy is 0.1144 and mean answer F1 is 0.145; gate G1 fails by a wide margin. Every result reported here is conditional on this reader. A transformer null head trained on 130,319 questions rather than our 4,736 might carry substantially more independent signal, in which case the relative contribution of the reader family would rise and the scope of our conclusion would narrow to weak-reader regimes. **Reproducing this measurement with a transformer reader is the most consequential outstanding step**, and the implementation is backend-agnostic to permit it.

**No method reached the error-prediction target.** Gate G2 fails; the best value is 0.6390 in-domain. Error prediction here is better than chance and useful for ranking, but it is not strong discrimination.

**No accuracy guarantee.** Gate G5 fails and every method sits at zero coverage for the 80% target. Calibration succeeded in the score-to-frequency sense (ECE 0.0120) and failed in the operating-point sense.

**Residual label risk.** After sibling masking, 2.39% of substantive unanswerable questions retain a plausible-answer string somewhere in the corpus — measured and small, but not zero. The string test is a proxy: presence does not establish answerability, and paraphrase could make a question answerable with no string match.

**Adversarially authored unanswerability.** SQuAD 2.0's unanswerable questions were written to appear plausible. They may be systematically easier or harder to detect than unanswerable questions arising naturally in deployment; this dataset cannot settle the matter.

**Corpus and language.** English Wikipedia only. No technical, multilingual or domain-specific evaluation.

**Retrieval granularity.** Paragraph-level retrieval avoids a chunking confound but means the results do not transfer directly to systems that must chunk long documents.

**Scale.** Single-machine CPU with an exact index over 5,637 passages. Behaviour at million-passage scale with an approximate index is untested.

**Transfer evidence is limited.** One transfer setting, moving corpus and reader instance together, without a paired test across corpora.

**Correlated features.** Leave-one-family-out bounds a family's contribution from below rather than partitioning credit.

**No deployment evidence.** No user study, no field trial, no production data. The system is a research prototype and is not claimed to be production-ready or superior to any commercial system.

## XIII. Conclusion

We measured how much error-prediction information retrieval-side and cross-evidence signals contribute beyond a reader-only calibrator in a CPU-only extractive question-answering pipeline. On 3,585 held-out SQuAD 2.0 questions across 36 unseen articles, with article-level splits and a passing negative control, the fused representation raised error-prediction ROC-AUC from 0.5242 to 0.6390 and improved AURC by 0.0345 with a paired bootstrap interval excluding zero. The direction held at three correctness thresholds and under transfer to a disjoint corpus, and the layer cost 1.96 ms. Ablation and permutation importance both attribute the contribution primarily to retrieval features and secondarily to agreement features, with the reader family's independent contribution yielding an interval containing zero.

The system is weak in absolute terms. Base accuracy is 0.1144, no method reached the pre-registered error-prediction target, and the intended 80% accuracy operating point is unreachable at any useful coverage; three of seven pre-registered gates failed. We report the incremental finding because it is measured, robust to several analysis choices and cheap to act on, and we report the failures because a stronger reader could change them and because a reader without them would be misled.

## Acknowledgment

The authors thank the Department of Computer Science and Engineering, AMC Engineering College, for institutional support. The SQuAD 2.0 dataset is used under the terms made available by its authors.

## References

[1] A. Kamath, R. Jia, and P. Liang, "Selective question answering under domain shift," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Online, Jul. 2020, pp. 5684–5696, doi: 10.18653/v1/2020.acl-main.503.

[2] P. Rajpurkar, R. Jia, and P. Liang, "Know what you don't know: Unanswerable questions for SQuAD," in *Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (ACL), Vol. 2: Short Papers*, Melbourne, Australia, Jul. 2018, pp. 784–789, doi: 10.18653/v1/P18-2124.

[3] S. Dhuliawala, L. Adolphs, R. Das, and M. Sachan, "Calibration of machine reading systems at scale," in *Findings Assoc. Comput. Linguistics: ACL 2022*, Dublin, Ireland, May 2022, pp. 1682–1693, doi: 10.18653/v1/2022.findings-acl.133.

[4] H. Joren *et al.*, "Sufficient context: A new lens on retrieval augmented generation systems," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2025. [Online]. Available: https://arxiv.org/abs/2411.06037

[5] "Answerability in retrieval-augmented open-domain question answering," arXiv:2403.01461, Mar. 2024. [Online]. Available: https://arxiv.org/abs/2403.01461

[6] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," *J. Mach. Learn. Res.*, vol. 11, pp. 1605–1641, May 2010.

[7] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4878–4887.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 70, Sydney, Australia, 2017, pp. 1321–1330.

[9] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, Edmonton, Canada, 2002, pp. 694–699, doi: 10.1145/775047.775151.

[10] P. Lewis, Y. Wu, L. Liu, P. Minervini, H. Küttler, A. Piktus, P. Stenetorp, and S. Riedel, "PAQ: 65 million probably-asked questions and what you can do with them," *Trans. Assoc. Comput. Linguistics*, vol. 9, 2021, doi: 10.1162/tacl_a_00415.

[11] S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman, "Indexing by latent semantic analysis," *J. Amer. Soc. Inf. Sci.*, vol. 41, no. 6, pp. 391–407, Sep. 1990.

[12] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retrieval*, vol. 3, no. 4, pp. 333–389, 2009, doi: 10.1561/1500000019.
