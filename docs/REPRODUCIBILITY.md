# Reproducibility

## Frozen configuration
| Item | Value |
|---|---|
| Seed | `20260814` (numpy, random, PYTHONHASHSEED) |
| Python | 3.10.12 |
| Corpus A | 120 articles from `train-v2.0.json`, 5,637 passages, 11,921 questions |
| Corpus B | all of `dev-v2.0.json`, 35 articles, 1,204 passages, 3,500 questions |
| Split | by article, 40/20/10/30 → 48 / 24 / 12 / 36 articles |
| Retrieval | k=5, SVD 256 dims, hybrid α=0.5 |
| Reader | max span 8 tokens, top 3 sentences, 400 negatives per positive |
| Reliability | HistGradientBoosting (200 iters, lr 0.1) + IsotonicRegression |
| Correctness | SQuAD F1 ≥ 0.5; sensitivity at 0.4 and 0.6 |
| Bootstrap | 1,000 resamples, clustered on **articles** |

Data hashes are in `data/raw/CHECKSUMS.txt`; config and table hashes are recorded inside
`results/RESULT_FREEZE.json`.

## Exact reproduction
```bash
pip install -r requirements.txt
cd data/raw && sha256sum -c CHECKSUMS.txt && cd ../..
for s in 01_build_corpus 02_run_pipeline 04_train_evaluate 05_ablation \
         06_domain_shift 07_latency 08_error_analysis 09_figures_and_freeze; do
  python3 experiments/$s.py
done
```
(`02_run_pipeline.py` takes an argument `A` or `B`; run both.)

## What must match
`results/RESULT_FREEZE.json` → `headline.P1_error_auc = 0.6389898…`,
`delta_aurc_P1_B5 = −0.03452…`, `gates.G0 = 0.48868…`.

Small deviations in **latency** (T7) are expected — they depend on the host CPU. Every other
number is deterministic given the seed.

## Known non-determinism
None in the statistical results. `HistGradientBoostingClassifier` and `TruncatedSVD` are both
seeded; the bootstrap uses `np.random.default_rng(seed)`.

## Freeze discipline
After `09_figures_and_freeze.py` writes `RESULT_FREEZE.json`, **no number is edited by hand.**
Any change re-runs the pipeline and produces a new freeze with a new timestamp.
