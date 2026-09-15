import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from veriqa.reliability.features import ALL, FAMILY, to_matrix
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_sixteen_features_in_three_families():
    assert len(ALL) == 16 and len(set(ALL)) == 16
    assert sum(len(v) for v in FAMILY.values()) == 16

def test_all_feature_values_finite_and_in_range():
    p = ROOT / "results/raw/predictions_A.jsonl"
    if not p.exists(): return
    rows = [json.loads(l)["features"] for i, l in zip(range(2000), open(p))]
    X = to_matrix(rows)
    assert np.isfinite(X).all(), "non-finite feature value"
    for name in ("d_span_prob", "d_null_score", "a_majority_frac",
                 "a_top2_string_sim", "r_lex_overlap", "a_answer_in_top2"):
        v = X[:, ALL.index(name)]
        assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9, f"{name} out of [0,1]"

def test_feature_order_is_stable():
    assert ALL[0] == "r_sim_top1" and ALL[-1] == "a_answer_in_top2"
