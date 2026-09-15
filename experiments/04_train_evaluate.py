"""E0 negative control, then E1 main comparison, E2 P1-vs-B5, E3 curves, E4 calibration.

E0 runs FIRST and halts the script if it fails, because every downstream number
is meaningless in the presence of leakage.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa.evaluation.squad_eval import is_correct, f1 as squad_f1, exact_match
from veriqa.evaluation.metrics import (error_auc, aurc, ece, risk_coverage_curve,
                                       selective_accuracy_at_coverage,
                                       coverage_at_target_accuracy,
                                       cluster_bootstrap_ci, paired_bootstrap_diff)
from veriqa.reliability.features import ALL, FAMILY, to_matrix
from veriqa.reliability.calibrator import RiskModel, choose_threshold
from veriqa.baselines.methods import (b1_always_answer, b2_span_prob, b3_retrieval_sim,
                                      b4_null_score, DESCRIPTIONS)

cfg = load_config(); set_all_seeds(cfg.seed)
THR = cfg.evaluation.f1_threshold
NB  = cfg.evaluation.bootstrap_n


def load(which):
    rows = [json.loads(l) for l in open(ROOT / f"results/raw/predictions_{which}.jsonl")]
    art = pickle.load(open(ROOT / f"models/artifacts_{which}.pkl", "rb"))
    part = {p: set(a) for p, a in art["art_split"].items()}
    by = {p: [r for r in rows if r["article"] in part[p]] for p in part}
    return rows, by, art


def labels(rows, thr=THR):
    return np.array([0.0 if is_correct(r["pred"], r["answers"], r["is_impossible"], thr)
                     else 1.0 for r in rows])


def feats(rows):
    return [r["features"] for r in rows]


def fit_pair(tr, cal, cols):
    m = RiskModel(max_iter=cfg.reliability.gbm_max_iter,
                  learning_rate=cfg.reliability.gbm_learning_rate, seed=cfg.seed)
    m.fit(to_matrix(feats(tr), cols), labels(tr))
    m.calibrate(to_matrix(feats(cal), cols), labels(cal))
    return m


def all_metrics(risk, correct, groups, tag):
    """`correct` must be 1.0 where the system was RIGHT. Passing the error
    array here silently inverts every metric, which is how the first run
    produced a below-chance AUC and an impossible AURC."""
    assert set(np.unique(correct)) <= {0.0, 1.0}, "correct must be 0/1"
    y = correct
    def _auc(idx): return error_auc(risk[idx], y[idx])
    def _aurc(idx): return aurc(risk[idx], y[idx])
    lo_a, hi_a = cluster_bootstrap_ci(None, groups, _auc, NB, cfg.seed)
    lo_r, hi_r = cluster_bootstrap_ci(None, groups, _aurc, NB, cfg.seed)
    d = {"method": tag, "n": len(y),
         "error_auc": error_auc(risk, y), "error_auc_lo": lo_a, "error_auc_hi": hi_a,
         "aurc": aurc(risk, y), "aurc_lo": lo_r, "aurc_hi": hi_r}
    for c in cfg.evaluation.coverages:
        d[f"sel_acc@{c}"] = selective_accuracy_at_coverage(risk, y, c)
    for t in cfg.evaluation.target_accuracies:
        d[f"cov@acc{t}"] = coverage_at_target_accuracy(risk, y, t)
    return d


if __name__ == "__main__":
    rows, by, art = load("A")
    tr, cal, te = by["rel_train"], by["rel_calib"], by["test"]
    y_te = labels(te)            # 1.0 = the reader was WRONG
    c_te = 1.0 - y_te            # 1.0 = the reader was CORRECT  <- metrics take this
    g_te = np.array([r["article"] for r in te])
    print(f"rel_train={len(tr):,}  rel_calib={len(cal):,}  test={len(te):,}")
    print(f"test error rate = {y_te.mean():.4f}  (correct = {1-y_te.mean():.4f})")

    # ---------------------------------------------------------------- E0 ----
    rng = np.random.default_rng(cfg.seed)
    y_tr_shuf = labels(tr).copy(); rng.shuffle(y_tr_shuf)
    m0 = RiskModel(seed=cfg.seed)
    m0.fit(to_matrix(feats(tr), ALL), y_tr_shuf)
    e0 = error_auc(m0.risk(to_matrix(feats(te), ALL), calibrated=False), 1.0 - y_te)
    print(f"\nE0 NEGATIVE CONTROL  ROC-AUC = {e0:.4f}   gate [0.45, 0.55]")
    e0_pass = 0.45 <= e0 <= 0.55
    print("E0:", "PASS" if e0_pass else "FAIL -- LEAKAGE SUSPECTED")
    json.dump({"e0_auc": e0, "pass": bool(e0_pass), "gate": [0.45, 0.55]},
              open(ROOT / "results/raw/E0_negative_control.json", "w"), indent=2)
    if not e0_pass:
        sys.exit("E0 failed; refusing to produce downstream numbers.")

    # ---------------------------------------------------------- E1 / E2 -----
    m_b5 = fit_pair(tr, cal, FAMILY["reader"])
    m_p1 = fit_pair(tr, cal, ALL)
    Xte_r = to_matrix(feats(te), FAMILY["reader"]); Xte_a = to_matrix(feats(te), ALL)
    risks = {
        "B1": b1_always_answer(feats(te)),
        "B2": b2_span_prob(feats(te)),
        "B3": b3_retrieval_sim(feats(te)),
        "B4": b4_null_score(feats(te)),
        "B5": m_b5.risk(Xte_r),
        "P1": m_p1.risk(Xte_a),
    }
    recs = [all_metrics(risks[k], c_te, g_te, k) for k in risks]
    T2 = pd.DataFrame(recs)
    T2["description"] = T2["method"].map(DESCRIPTIONS)
    T2.to_csv(ROOT / "results/tables/T2_main_comparison.csv", index=False)
    print("\n--- E1 MAIN COMPARISON (test) ---")
    print(T2[["method", "error_auc", "error_auc_lo", "error_auc_hi", "aurc",
              "aurc_lo", "aurc_hi", "sel_acc@0.5", "sel_acc@0.7"]].round(4).to_string(index=False))

    mean_d, lo, hi = paired_bootstrap_diff(risks["P1"], risks["B5"], c_te, g_te,
                                           aurc, NB, cfg.seed)
    mean_d4, lo4, hi4 = paired_bootstrap_diff(risks["P1"], risks["B4"], c_te, g_te,
                                              aurc, NB, cfg.seed)
    e2 = {"delta_aurc_P1_minus_B5": mean_d, "lo": lo, "hi": hi,
          "excludes_zero": bool(hi < 0 or lo > 0),
          "delta_aurc_P1_minus_B4": mean_d4, "lo_b4": lo4, "hi_b4": hi4,
          "excludes_zero_b4": bool(hi4 < 0 or lo4 > 0)}
    json.dump(e2, open(ROOT / "results/raw/E2_p1_vs_b5.json", "w"), indent=2)
    print(f"\n--- E2 P1 vs B5 --- dAURC = {mean_d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"(negative favours P1); excludes 0: {e2['excludes_zero']}")
    print(f"    P1 vs B4         dAURC = {mean_d4:+.4f} [{lo4:+.4f}, {hi4:+.4f}]; "
          f"excludes 0: {e2['excludes_zero_b4']}")

    # ------------------------------------------------------------- E3 -------
    curves = {}
    for k, r in risks.items():
        cov, sel = risk_coverage_curve(r, c_te)
        curves[k] = {"coverage": cov.tolist(), "selective_risk": sel.tolist()}
    json.dump(curves, open(ROOT / "results/raw/E3_risk_coverage.json", "w"))

    # ------------------------------------------------------------- E4 -------
    raw_p1 = m_p1.risk(Xte_a, calibrated=False)
    cal_p1 = risks["P1"]
    thr, acc_cal, cov_cal = choose_threshold(
        m_p1.risk(to_matrix(feats(cal), ALL)), 1 - labels(cal),
        cfg.reliability.target_accuracy)
    answered = cal_p1 <= thr
    realised = float((1 - y_te)[answered].mean()) if answered.any() else float("nan")
    T3 = pd.DataFrame([{
        "ece_before_isotonic": ece(raw_p1, 1 - y_te),
        "ece_after_isotonic": ece(cal_p1, 1 - y_te),
        "target_accuracy": cfg.reliability.target_accuracy,
        "threshold_from_calibration": thr,
        "accuracy_on_calibration": acc_cal, "coverage_on_calibration": cov_cal,
        "accuracy_realised_on_test": realised,
        "coverage_realised_on_test": float(answered.mean()),
        "realisation_gap": realised - cfg.reliability.target_accuracy,
    }])
    T3.to_csv(ROOT / "results/tables/T3_calibration.csv", index=False)
    print("\n--- E4 CALIBRATION ---"); print(T3.round(4).to_string(index=False))

    # --------------------------------------------------- T1 dataset stats ---
    def blk(rs, name):
        yy = labels(rs); imp = np.array([r["is_impossible"] for r in rs])
        ansr = [r for r in rs if not r["is_impossible"]]
        return {"partition": name, "questions": len(rs),
                "articles": len({r["article"] for r in rs}),
                "unanswerable": int(imp.sum()),
                "unanswerable_frac": float(imp.mean()),
                "gold_in_topk": float(np.mean([r["gold_retrieved"] for r in rs])),
                "reader_EM": float(np.mean([exact_match(r["pred"], r["answers"])
                                            for r in ansr])) if ansr else float("nan"),
                "reader_F1": float(np.mean([squad_f1(r["pred"], r["answers"])
                                            for r in ansr])) if ansr else float("nan"),
                "error_rate": float(yy.mean())}
    T1 = pd.DataFrame([blk(tr, "rel_train"), blk(cal, "rel_calib"), blk(te, "test")])
    T1.to_csv(ROOT / "results/tables/T1_dataset.csv", index=False)
    print("\n--- T1 DATASET / PIPELINE ---"); print(T1.round(4).to_string(index=False))
    pickle.dump({"m_b5": m_b5, "m_p1": m_p1, "threshold": thr},
                open(ROOT / "models/reliability_A.pkl", "wb"))
    print("\nE0-E4 complete")
