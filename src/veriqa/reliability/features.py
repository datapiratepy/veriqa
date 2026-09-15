"""The 16 reliability features, in three families.

Every feature is computed from artefacts the pipeline has already produced.
Nothing here triggers an additional model pass, which is what keeps the
reliability layer cheap enough to be reported as a deployment cost.
"""
from __future__ import annotations
import difflib
import numpy as np
from ..evaluation.squad_eval import normalize_answer
from ..reader.spans import STOP

RETRIEVAL = ["r_sim_top1", "r_sim_margin", "r_sim_mean_k", "r_sim_entropy",
             "r_n_above_tau", "r_lex_overlap", "r_q_len"]
READER = ["d_span_prob", "d_null_score", "d_gap", "d_span_len", "d_start_end_entropy"]
AGREEMENT = ["a_n_nonnull", "a_top2_string_sim", "a_majority_frac", "a_answer_in_top2"]
ALL = RETRIEVAL + READER + AGREEMENT
FAMILY = {"retrieval": RETRIEVAL, "reader": READER, "agreement": AGREEMENT}
assert len(ALL) == 16


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p - p.min() + 1e-9
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def retrieval_features(retrieved, question: str, passages, tau: float) -> dict:
    h = np.asarray(retrieved.hybrid, dtype=np.float64)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        h = np.zeros(1)
    wide = np.asarray(retrieved.all_hybrid, dtype=np.float64)
    wide = wide[np.isfinite(wide)]
    import re
    qtok = {w for w in re.findall(r"\w+", question.lower())
            if w not in STOP and len(w) > 1}
    top_txt = set(re.findall(r"\w+", passages[0].text.lower())) if passages else set()
    lex = len(qtok & top_txt) / len(qtok) if qtok else 0.0
    return {
        "r_sim_top1": float(h[0]),
        "r_sim_margin": float(h[0] - h[1]) if len(h) > 1 else float(h[0]),
        "r_sim_mean_k": float(h.mean()),
        "r_sim_entropy": _entropy(wide if len(wide) > 1 else h),
        "r_n_above_tau": float((h > tau).sum()),
        "r_lex_overlap": float(lex),
        "r_q_len": float(len(question.split())),
    }


def reader_features(out) -> dict:
    return {
        "d_span_prob": float(out.span_prob),
        "d_null_score": float(out.null_score),
        "d_gap": float(out.span_prob - out.null_score),
        "d_span_len": float(out.span_len),
        "d_start_end_entropy": float(out.span_entropy),
    }


def agreement_features(out, passages) -> dict:
    per = out.per_passage
    answers = [normalize_answer(a) for _, a, _, _ in per]
    nonnull = [a for a in answers if a]
    n_nonnull = float(len(nonnull))
    if len(answers) >= 2 and answers[0] and answers[1]:
        sim = difflib.SequenceMatcher(None, answers[0], answers[1]).ratio()
    else:
        sim = 0.0
    if nonnull:
        counts = {a: nonnull.count(a) for a in set(nonnull)}
        majority = max(counts.values()) / len(nonnull)
    else:
        majority = 0.0
    top = normalize_answer(out.answer)
    second_text = passages[1].text.lower() if len(passages) > 1 else ""
    in_top2 = float(bool(top) and top in second_text)
    return {
        "a_n_nonnull": n_nonnull,
        "a_top2_string_sim": float(sim),
        "a_majority_frac": float(majority),
        "a_answer_in_top2": in_top2,
    }


def extract_all(retrieved, out, question, passages, tau) -> dict:
    f = {}
    f.update(retrieval_features(retrieved, question, passages, tau))
    f.update(reader_features(out))
    f.update(agreement_features(out, passages))
    assert set(f) == set(ALL), "feature set drifted from the pre-registered 16"
    return f


def to_matrix(rows, cols=None) -> np.ndarray:
    cols = cols or ALL
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
