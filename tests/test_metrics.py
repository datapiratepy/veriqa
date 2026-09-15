import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from veriqa.evaluation.metrics import (aurc, error_auc, ece,
                                       selective_accuracy_at_coverage,
                                       coverage_at_target_accuracy,
                                       risk_coverage_curve)

# Hand-computed fixture -------------------------------------------------------
# risk    : 0.1  0.2  0.3  0.4  0.5
# correct :  1    1    0    1    0
RISK = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
COR = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

def test_risk_coverage_curve_by_hand():
    cov, sel = risk_coverage_curve(RISK, COR)
    # cumulative errors 0,0,1,1,2 over n=1..5
    assert np.allclose(sel, [0, 0, 1/3, 1/4, 2/5])
    assert np.allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])

def test_aurc_by_hand():
    expected = (0 + 0 + 1/3 + 1/4 + 2/5) / 5     # = 0.19666...
    assert abs(aurc(RISK, COR) - expected) < 1e-12

def test_selective_accuracy_by_hand():
    # coverage 0.6 -> 3 lowest-risk items, 2 of 3 correct
    assert abs(selective_accuracy_at_coverage(RISK, COR, 0.6) - 2/3) < 1e-12
    assert abs(selective_accuracy_at_coverage(RISK, COR, 1.0) - 3/5) < 1e-12

def test_coverage_at_target_by_hand():
    # accuracies by coverage: 1.0, 1.0, 0.667, 0.75, 0.6
    assert abs(coverage_at_target_accuracy(RISK, COR, 0.75) - 0.8) < 1e-12
    assert abs(coverage_at_target_accuracy(RISK, COR, 1.0) - 0.4) < 1e-12

def test_error_auc_by_hand():
    # errors sit at risk {0.3, 0.5}; non-errors at {0.1, 0.2, 0.4}.
    # Of the 6 error/non-error pairs, only (0.3 vs 0.4) is mis-ordered -> 5/6.
    assert abs(error_auc(RISK, COR) - 5/6) < 1e-9
    assert abs(error_auc(-RISK, COR) - 1/6) < 1e-9        # reversing flips it

def test_error_auc_perfect_ranking():
    risk = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    cor  = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert abs(error_auc(risk, cor) - 1.0) < 1e-9

def test_ece_perfectly_calibrated_is_zero():
    p = np.array([0.0]*50 + [1.0]*50)
    correct = np.array([1.0]*50 + [0.0]*50)
    assert ece(p, correct) < 1e-9

def test_ece_worst_case_is_one():
    p = np.array([1.0]*50 + [0.0]*50)
    correct = np.array([1.0]*50 + [0.0]*50)   # says wrong when right, and vice versa
    assert abs(ece(p, correct) - 1.0) < 1e-9
