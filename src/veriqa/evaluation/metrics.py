"""Selective-prediction metrics. Each is verified against a hand-computed case
in tests/test_metrics.py.

Convention: `risk` is a score where HIGHER means more likely to be wrong.
`correct` is 1.0 when the system's output was correct.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def error_auc(risk, correct):
    """ROC-AUC for predicting ERROR from the risk score. 0.5 = chance."""
    y = 1.0 - np.asarray(correct, float)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(risk, float)))


def risk_coverage_curve(risk, correct):
    """Return (coverage, selective_risk) with coverage from 1/n to 1."""
    risk, correct = np.asarray(risk, float), np.asarray(correct, float)
    order = np.argsort(risk, kind="mergesort")
    c = correct[order]
    n = len(c)
    cum_err = np.cumsum(1.0 - c)
    cov = np.arange(1, n + 1) / n
    sel_risk = cum_err / np.arange(1, n + 1)
    return cov, sel_risk


def aurc(risk, correct):
    """Area under the risk-coverage curve. LOWER IS BETTER."""
    _, sel = risk_coverage_curve(risk, correct)
    return float(sel.mean())


def selective_accuracy_at_coverage(risk, correct, coverage):
    risk, correct = np.asarray(risk, float), np.asarray(correct, float)
    n = max(1, int(round(coverage * len(risk))))
    order = np.argsort(risk, kind="mergesort")[:n]
    return float(correct[order].mean())


def coverage_at_target_accuracy(risk, correct, target):
    """Largest coverage whose selective accuracy still meets `target`."""
    cov, sel = risk_coverage_curve(risk, correct)
    acc = 1.0 - sel
    ok = np.where(acc >= target)[0]
    return float(cov[ok.max()]) if len(ok) else 0.0


def ece(prob_error, correct, n_bins=10):
    """Expected calibration error of a predicted error probability."""
    p = np.asarray(prob_error, float)
    y = 1.0 - np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(total)


def cluster_bootstrap_ci(values, groups, stat_fn, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap CI resampling GROUPS (articles), not individual questions.

    Questions from one article are correlated; resampling questions would give
    intervals that are far too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        try:
            v = stat_fn(idx)
        except Exception:
            continue
        if np.isfinite(v):
            out.append(v)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def paired_bootstrap_diff(risk_a, risk_b, correct, groups, metric_fn,
                          n_boot=1000, seed=0):
    """CI for metric(a) - metric(b) on the SAME resampled articles."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        d = metric_fn(risk_a[idx], correct[idx]) - metric_fn(risk_b[idx], correct[idx])
        if np.isfinite(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)
