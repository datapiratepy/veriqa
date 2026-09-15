# Architecture

```
        user question
              │
   ┌──────────▼───────────┐
   │  Streamlit UI  /  FastAPI  (/ask /health /evaluate)   │
   └──────────┬───────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │ HYBRID RETRIEVAL                                     │
   │  TF-IDF ─► TruncatedSVD(256) ─► exact inner product  │
   │  BM25Okapi                                           │
   │  min-max normalise, fuse α=0.5, sibling mask (D-002) │
   └──────────┬──────────────────────────────────────────┘
              │  top-k passages + dense/bm25/hybrid scores
   ┌──────────▼──────────────────────────────────────────┐
   │ TWO-STAGE EXTRACTIVE READER                          │
   │  1. rank sentences by question overlap, keep top 3   │
   │  2. enumerate n-grams ≤8, drop stopword/punct edges  │
   │  3. GBM span scorer over 20 span features            │
   │  4. GBM null head → P(no answer in this passage)     │
   └──────────┬──────────────────────────────────────────┘
              │  span, span_prob, null_score, entropy, per-passage answers
   ┌──────────▼──────────────────────────────────────────┐
   │ ★ RELIABILITY LAYER — the contribution               │
   │   retrieval (7) + reader (5) + agreement (4) = 16    │
   │   HistGradientBoosting → isotonic → P(error)         │
   │   threshold chosen on rel_calib only                 │
   └──────────┬──────────────────────────────────────────┘
              │
      risk > τ ? ─── yes ──► ABSTAIN + reason
              │
              no ──► answer + passage + highlighted span + risk
```

## Module map

| Path | Responsibility | Owner |
|---|---|---|
| `ingestion/squad.py` | SQuAD loading, corpus construction, sampling | Harsh Ranjan |
| `retrieval/hybrid.py` | LSA index, BM25, fusion, sibling masking | Harsh Ranjan |
| `reader/spans.py` | candidate generation, 20 vectorised span features | Harsh Kamat |
| `reader/extractive.py` | span scorer, null head, `read()` | Harsh Kamat |
| `reliability/features.py` | the 16 reliability features | Harsh Kamat |
| `reliability/calibrator.py` | risk model, isotonic, threshold selection | Harsh Kamat |
| `baselines/methods.py` | B1–B5 | Harsh Kamat |
| `evaluation/splits.py` | **the only place splits are created** | Manish Kumar |
| `evaluation/metrics.py` | AUC, AURC, ECE, selective accuracy, bootstrap | Manish Kumar |
| `evaluation/squad_eval.py` | official normalisation, EM, F1, correctness | Manish Kumar |
| `service.py`, `api/`, `app/` | query service, REST, UI | Manish Kumar |

## Design constraints that shaped it

- **No pretrained weights** (D-009) → LSA instead of sentence-transformers, feature-based reader
  instead of DistilBERT.
- **Every reliability feature reads artefacts the pipeline already produced** → no second model
  pass → layer costs 1.96 ms rather than doubling inference.
- **Exact retrieval index, not approximate** → no approximation error confounding the
  retrieval-signal measurements.
- **Splits created in exactly one function** → leakage is testable, and is tested.
