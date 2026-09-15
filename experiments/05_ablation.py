"""E5 ablation (7 pre-registered configs) + permutation importance cross-check."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa.evaluation.metrics import aurc, error_auc, paired_bootstrap_diff, cluster_bootstrap_ci
from veriqa.reliability.features import ALL, FAMILY, to_matrix
from veriqa.reliability.calibrator import RiskModel
sys.path.insert(0, str(ROOT / "experiments"))
from importlib import import_module
te_mod = import_module("04_train_evaluate") if False else None
from veriqa.evaluation.squad_eval import is_correct

cfg = load_config(); set_all_seeds(cfg.seed)

def load():
    rows = [json.loads(l) for l in open(ROOT / "results/raw/predictions_A.jsonl")]
    art = pickle.load(open(ROOT / "models/artifacts_A.pkl", "rb"))
    part = {p: set(a) for p, a in art["art_split"].items()}
    return {p: [r for r in rows if r["article"] in part[p]] for p in part}

def lab(rows):
    return np.array([0.0 if is_correct(r["pred"], r["answers"], r["is_impossible"],
                                       cfg.evaluation.f1_threshold) else 1.0 for r in rows])

CONFIGS = {
    "Full (R+D+A)":  ALL,
    "- Retrieval":   FAMILY["reader"] + FAMILY["agreement"],
    "- Reader":      FAMILY["retrieval"] + FAMILY["agreement"],
    "- Agreement":   FAMILY["retrieval"] + FAMILY["reader"],
    "Retrieval only": FAMILY["retrieval"],
    "Reader only":    FAMILY["reader"],
    "Agreement only": FAMILY["agreement"],
}

if __name__ == "__main__":
    by = load(); tr, cal, te = by["rel_train"], by["rel_calib"], by["test"]
    y_tr, y_cal, y_te = lab(tr), lab(cal), lab(te)
    c_te = 1.0 - y_te
    g_te = np.array([r["article"] for r in te])
    ftr = [r["features"] for r in tr]; fcal = [r["features"] for r in cal]
    fte = [r["features"] for r in te]

    risks, recs = {}, []
    for name, cols in CONFIGS.items():
        m = RiskModel(max_iter=cfg.reliability.gbm_max_iter,
                      learning_rate=cfg.reliability.gbm_learning_rate, seed=cfg.seed)
        m.fit(to_matrix(ftr, cols), y_tr); m.calibrate(to_matrix(fcal, cols), y_cal)
        r = m.risk(to_matrix(fte, cols)); risks[name] = r
        lo, hi = cluster_bootstrap_ci(None, g_te, lambda i: aurc(r[i], c_te[i]),
                                      cfg.evaluation.bootstrap_n, cfg.seed)
        recs.append({"config": name, "n_features": len(cols),
                     "error_auc": error_auc(r, c_te), "aurc": aurc(r, c_te),
                     "aurc_lo": lo, "aurc_hi": hi})
    full = risks["Full (R+D+A)"]
    for rec in recs:
        if rec["config"] == "Full (R+D+A)":
            rec.update(d_aurc_vs_full=0.0, d_lo=0.0, d_hi=0.0, excludes_zero=False)
            continue
        d, lo, hi = paired_bootstrap_diff(risks[rec["config"]], full, c_te, g_te,
                                          aurc, cfg.evaluation.bootstrap_n, cfg.seed)
        rec.update(d_aurc_vs_full=d, d_lo=lo, d_hi=hi,
                   excludes_zero=bool(hi < 0 or lo > 0))
    T4 = pd.DataFrame(recs)
    T4.to_csv(ROOT / "results/tables/T4_ablation.csv", index=False)
    print("--- E5 ABLATION (dAURC positive = worse than Full) ---")
    print(T4.round(4).to_string(index=False))

    # permutation importance on the full model, as an independent attribution
    m = RiskModel(seed=cfg.seed).fit(to_matrix(ftr, ALL), y_tr)
    m.calibrate(to_matrix(fcal, ALL), y_cal)
    Xte = to_matrix(fte, ALL); base = aurc(m.risk(Xte), c_te)
    rng = np.random.default_rng(cfg.seed); imp = []
    for j, name in enumerate(ALL):
        deltas = []
        for _ in range(10):
            Xp = Xte.copy(); rng.shuffle(Xp[:, j])
            deltas.append(aurc(m.risk(Xp), c_te) - base)
        fam = next(k for k, v in FAMILY.items() if name in v)
        imp.append({"feature": name, "family": fam,
                    "delta_aurc_when_shuffled": float(np.mean(deltas)),
                    "std": float(np.std(deltas))})
    T5 = pd.DataFrame(imp).sort_values("delta_aurc_when_shuffled", ascending=False)
    T5.to_csv(ROOT / "results/tables/T5_permutation_importance.csv", index=False)
    print("\n--- PERMUTATION IMPORTANCE (higher = more load-bearing) ---")
    print(T5.round(5).to_string(index=False))
