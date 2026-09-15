"""Risk model: gradient boosting -> isotonic calibration -> threshold gate."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression


class RiskModel:
    """Predicts P(the reader's answer is wrong)."""

    def __init__(self, max_iter=200, learning_rate=0.1, seed=0):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, random_state=seed)
        self.iso = None

    def fit(self, X, y_error):
        self.clf.fit(X, y_error)
        return self

    def calibrate(self, X_cal, y_cal):
        raw = self.clf.predict_proba(X_cal)[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(raw, y_cal)
        return self

    def risk(self, X, calibrated=True):
        raw = self.clf.predict_proba(X)[:, 1]
        if calibrated and self.iso is not None:
            return np.clip(self.iso.predict(raw), 0.0, 1.0)
        return raw


def choose_threshold(risk_cal, correct_cal, target_accuracy: float):
    """Lowest-risk threshold on CALIBRATION data achieving the target accuracy.

    Returns (threshold, achieved_accuracy, coverage). If the target is
    unreachable at any coverage, returns the threshold maximising accuracy.
    """
    risk_cal = np.asarray(risk_cal, float)
    correct_cal = np.asarray(correct_cal, float)
    order = np.argsort(risk_cal)
    acc_sorted = correct_cal[order]
    cum = np.cumsum(acc_sorted) / np.arange(1, len(acc_sorted) + 1)
    ok = np.where(cum >= target_accuracy)[0]
    if len(ok) == 0:
        j = int(np.argmax(cum))
        return float(risk_cal[order][j]), float(cum[j]), float((j + 1) / len(cum))
    j = int(ok.max())
    return float(risk_cal[order][j]), float(cum[j]), float((j + 1) / len(cum))
