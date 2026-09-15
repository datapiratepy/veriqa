"""E6 domain shift: reliability model fitted on Corpus A, evaluated on Corpus B.

Corpus A is sampled from SQuAD 2.0 train articles; Corpus B is the SQuAD 2.0 dev
articles. Article titles are disjoint (verified), so this is a genuine transfer
test. Note it is a transfer of the RELIABILITY model across BOTH a new corpus and
a separately-fitted reader instance -- the realistic deployment condition. An
in-domain B model is reported alongside as the ceiling.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa.evaluation.metrics import aurc, error_auc, ece, cluster_bootstrap_ci
from veriqa.evaluation.squad_eval import is_correct
from veriqa.reliability.features import ALL, FAMILY, to_matrix
from veriqa.reliability.calibrator import RiskModel

cfg = load_config(); set_all_seeds(cfg.seed)

def load(w):
    rows = [json.loads(l) for l in open(ROOT / f"results/raw/predictions_{w}.jsonl")]
    art = pickle.load(open(ROOT / f"models/artifacts_{w}.pkl", "rb"))
    part = {p: set(a) for p, a in art["art_split"].items()}
    return {p: [r for r in rows if r["article"] in part[p]] for p in part}

def lab(rs):
    return np.array([0.0 if is_correct(r["pred"], r["answers"], r["is_impossible"],
                                       cfg.evaluation.f1_threshold) else 1.0 for r in rs])

def fit(tr, cal, cols):
    m = RiskModel(max_iter=cfg.reliability.gbm_max_iter,
                  learning_rate=cfg.reliability.gbm_learning_rate, seed=cfg.seed)
    m.fit(to_matrix([r["features"] for r in tr], cols), lab(tr))
    m.calibrate(to_matrix([r["features"] for r in cal], cols), lab(cal))
    return m

if __name__ == "__main__":
    A, B = load("A"), load("B")
    bte = B["test"]; c_b = 1.0 - lab(bte); g_b = np.array([r["article"] for r in bte])
    Xb = {"P1": to_matrix([r["features"] for r in bte], ALL),
          "B5": to_matrix([r["features"] for r in bte], FAMILY["reader"])}
    ate = A["test"]; c_a = 1.0 - lab(ate); g_a = np.array([r["article"] for r in ate])
    Xa = {"P1": to_matrix([r["features"] for r in ate], ALL),
          "B5": to_matrix([r["features"] for r in ate], FAMILY["reader"])}

    recs = []
    for tag, cols in (("P1", ALL), ("B5", FAMILY["reader"])):
        mA = fit(A["rel_train"], A["rel_calib"], cols)
        mB = fit(B["rel_train"], B["rel_calib"], cols)
        for name, m, X, c, g in (
                ("A -> A (in-domain)", mA, Xa[tag], c_a, g_a),
                ("A -> B (transfer)",  mA, Xb[tag], c_b, g_b),
                ("B -> B (in-domain ceiling)", mB, Xb[tag], c_b, g_b)):
            r = m.risk(X)
            lo, hi = cluster_bootstrap_ci(None, g, lambda i: aurc(r[i], c[i]),
                                          cfg.evaluation.bootstrap_n, cfg.seed)
            recs.append({"method": tag, "setting": name, "n": len(c),
                         "error_auc": error_auc(r, c), "aurc": aurc(r, c),
                         "aurc_lo": lo, "aurc_hi": hi, "ece": ece(r, c),
                         "base_error_rate": float(1 - c.mean())})
    T6 = pd.DataFrame(recs)
    T6.to_csv(ROOT / "results/tables/T6_domain_shift.csv", index=False)
    print("--- E6 DOMAIN SHIFT ---"); print(T6.round(4).to_string(index=False))
