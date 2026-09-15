# Reader Backend: why this is not a transformer

**Decision ID:** D-009 · **Status:** binding on all reported results

## What happened

The design specified `distilbert-base-cased-distilled-squad` (or an equivalent SQuAD 2.0
transformer) as the reader and `all-MiniLM-L6-v2` as the embedder. Neither could be obtained.
Every route to Hugging Face is blocked in the build environment:

```
https://huggingface.co            403 blocked-by-allowlist
https://huggingface.co/api/models 403
https://cdn-lfs.huggingface.co    403
https://hf.co                     403
```

Kaggle, UCI, Zenodo and `raw.githubusercontent.com` are likewise blocked. `github.com` over
`git clone` and PyPI are reachable, which is how SQuAD 2.0 itself was obtained
(`rajpurkar/SQuAD-explorer`, SHA-256 recorded in `data/raw/CHECKSUMS.txt`).

## The options, and why we chose this one

1. **Ship code with empty result tables.** Honest, but produces no experiment and no paper.
2. **Fabricate plausible numbers.** Forbidden, and never considered.
3. **Substitute components that need no pretrained weights, and say so everywhere.** Chosen.

## What was substituted

| Designed | Built | Class of method |
|---|---|---|
| `all-MiniLM-L6-v2` sentence embeddings | TF-IDF + truncated SVD (256 dims) | Latent semantic analysis |
| DistilBERT SQuAD 2.0 extractive reader | Two-stage feature-based span scorer + trained null head | Classical feature-based QA |

Both substitutes are established methods, not improvisations, and both preserve the *interface*
the research question depends on: the reader still emits a best span, a span probability, a span
distribution entropy and an explicit null-answer score, so all sixteen reliability features are
computed exactly as specified.

## The cost, stated plainly

The reader is much weaker than a transformer would be. On the frozen test set:

- mean answer F1 on answerable questions: **0.145**
- overall base accuracy: **0.114**
- pre-registered gate G1 (accuracy ≥ 0.45): **FAILED**

Every result in the paper is conditional on this reader. The comparative finding — that
retrieval features improve error prediction over reader features — was measured in a regime
where the reader is weak, and it may not hold with a strong one. The manuscript says this in
the abstract, in Section XI and in the conclusion.

## What the team should do

The pipeline is backend-agnostic. `src/veriqa/reader/extractive.py` exposes `read()` returning
`ReadOut(answer, span_prob, null_score, span_len, span_entropy, passage_pid, per_passage)`.
A transformer reader that fills the same structure is a drop-in replacement, and
`src/veriqa/retrieval/hybrid.py` can take neural embeddings the same way.

**Re-running the full experiment with a transformer reader on a machine with Hugging Face
access is the single highest-value remaining task.** It changes every absolute number and may
change the comparative conclusion. That is the point.
