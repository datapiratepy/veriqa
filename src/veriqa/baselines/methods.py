"""Baselines B1-B5 and the proposed method P1.

All consume identical pipeline outputs on identical splits; only the
abstain/answer signal differs. Higher score = more likely to be WRONG.
"""
from __future__ import annotations
import numpy as np
from ..reliability.features import FAMILY, to_matrix
from ..reliability.calibrator import RiskModel

DESCRIPTIONS = {
    "B1": "Always answer (no abstention) - floor",
    "B2": "Reader span-probability threshold",
    "B3": "Retrieval top-1 similarity threshold",
    "B4": "Reader null-answer score (the model's own trained no-answer head)",
    "B5": "Reader-only calibrator (Kamath, Jia & Liang 2020 analogue)",
    "P1": "Proposed: fused calibrator over retrieval + reader + agreement features",
}


def b1_always_answer(rows):
    return np.zeros(len(rows))                      # constant risk -> no ordering


def b2_span_prob(rows):
    return -to_matrix(rows, ["d_span_prob"])[:, 0]  # low confidence = high risk


def b3_retrieval_sim(rows):
    return -to_matrix(rows, ["r_sim_top1"])[:, 0]


def b4_null_score(rows):
    return to_matrix(rows, ["d_null_score"])[:, 0]  # already "probability of no answer"


def fit_calibrator(rows_tr, y_tr, rows_cal, y_cal, cols, seed, **kw):
    """Fit a RiskModel on `cols` only. Used for B5 (reader-only) and P1 (all)."""
    m = RiskModel(seed=seed, **kw)
    m.fit(to_matrix(rows_tr, cols), y_tr)
    m.calibrate(to_matrix(rows_cal, cols), y_cal)
    return m


READER_ONLY = FAMILY["reader"]
