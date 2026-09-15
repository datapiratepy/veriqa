"""Figures from frozen tables, then RESULT_FREEZE.json. Nothing is recomputed by hand."""
import sys, json, hashlib, platform, subprocess, datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
cfg = load_config()
T = lambda n: pd.read_csv(ROOT / f"results/tables/{n}.csv")
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})

# F3 risk-coverage
curves = json.load(open(ROOT / "results/raw/E3_risk_coverage.json"))
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for k in ["B1", "B2", "B3", "B4", "B5", "P1"]:
    c = curves[k]; ls = "-" if k == "P1" else "--"
    lw = 2.0 if k == "P1" else 1.0
    ax.plot(c["coverage"], c["selective_risk"], ls, lw=lw, label=k)
ax.set_xlabel("coverage"); ax.set_ylabel("selective risk (error rate among answered)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
fig.savefig(FIG / "F3_risk_coverage.png"); plt.close(fig)

# F5 ablation
t4 = T("T4_ablation")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
d = t4[t4.config != "Full (R+D+A)"].sort_values("d_aurc_vs_full")
err = np.vstack([d.d_aurc_vs_full - d.d_lo, d.d_hi - d.d_aurc_vs_full])
ax.barh(d.config, d.d_aurc_vs_full, xerr=err, color=["#c0392b" if e else "#95a5a6"
        for e in d.excludes_zero], capsize=3)
ax.axvline(0, color="k", lw=.8); ax.set_xlabel(r"$\Delta$AURC vs Full (higher = worse)")
fig.savefig(FIG / "F5_ablation.png"); plt.close(fig)

# F4 reliability diagram
import pickle
from veriqa.reliability.features import ALL, to_matrix
from veriqa.evaluation.squad_eval import is_correct
rows = [json.loads(l) for l in open(ROOT / "results/raw/predictions_A.jsonl")]
artf = pickle.load(open(ROOT / "models/artifacts_A.pkl", "rb"))
te = [r for r in rows if r["article"] in set(artf["art_split"]["test"])]
rel = pickle.load(open(ROOT / "models/reliability_A.pkl", "rb"))
X = to_matrix([r["features"] for r in te], ALL)
y_err = np.array([0.0 if is_correct(r["pred"], r["answers"], r["is_impossible"],
                  cfg.evaluation.f1_threshold) else 1.0 for r in te])
raw, calp = rel["m_p1"].risk(X, calibrated=False), rel["m_p1"].risk(X)
fig, ax = plt.subplots(figsize=(3.4, 3.2))
for p, lbl, mk in ((raw, "before isotonic", "o"), (calp, "after isotonic", "s")):
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for i in range(10):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() > 20: xs.append(p[m].mean()); ys.append(y_err[m].mean())
    ax.plot(xs, ys, mk + "-", ms=4, label=lbl)
ax.plot([0, 1], [0, 1], "k:", lw=.8, label="perfect")
ax.set_xlabel("predicted P(error)"); ax.set_ylabel("observed error rate")
ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.savefig(FIG / "F4_reliability_diagram.png"); plt.close(fig)

# F2 feature separation
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, f in zip(axes, ["r_lex_overlap", "a_answer_in_top2", "d_span_prob"]):
    v = np.array([r["features"][f] for r in te])
    ax.hist(v[y_err == 0], bins=20, alpha=.6, label="correct", density=True)
    ax.hist(v[y_err == 1], bins=20, alpha=.6, label="error", density=True)
    ax.set_title(f, fontsize=8); ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7)
fig.savefig(FIG / "F2_feature_separation.png"); plt.close(fig)

# F6 error taxonomy
t9 = T("T9_error_taxonomy")
fig, ax = plt.subplots(figsize=(4.8, 2.4))
ax.barh(t9.category.str.slice(0, 42), t9.fraction, color="#34495e")
ax.set_xlabel("fraction of test questions"); ax.tick_params(labelsize=7)
fig.savefig(FIG / "F6_error_taxonomy.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))

# ------------------------------- GATES + FREEZE ------------------------------
t2 = T("T2_main_comparison").set_index("method")
e0 = json.load(open(ROOT / "results/raw/E0_negative_control.json"))
e2 = json.load(open(ROOT / "results/raw/E2_p1_vs_b5.json"))
t1 = T("T1_dataset").set_index("partition"); t3 = T("T3_calibration").iloc[0]
t7 = T("T7_latency").set_index("stage"); t6 = T("T6_domain_shift")
acc = 1 - float(t1.loc["test", "error_rate"])
best_auc = float(t2["error_auc"].max())
rl = float(t7.loc["RELIABILITY LAYER TOTAL (features + gate)", "median_ms"])
gates = {
 "G0_negative_control_in_0.45_0.55": {"value": e0["e0_auc"], "pass": e0["pass"]},
 "G1_base_reader_accuracy_ge_0.45": {"value": acc, "pass": bool(acc >= 0.45)},
 "G2_any_method_error_auc_ge_0.70": {"value": best_auc, "pass": bool(best_auc >= 0.70)},
 "G3_P1_AURC_lt_B5_CI_excludes_0": {"value": e2["delta_aurc_P1_minus_B5"],
    "ci": [e2["lo"], e2["hi"]], "pass": bool(e2["excludes_zero"] and e2["hi"] < 0)},
 "G4_P1_AURC_lt_B4_CI_excludes_0": {"value": e2["delta_aurc_P1_minus_B4"],
    "ci": [e2["lo_b4"], e2["hi_b4"]], "pass": bool(e2["excludes_zero_b4"] and e2["hi_b4"] < 0)},
 "G5_ECE_lt_0.10_and_target_within_3pts": {
    "ece": float(t3.ece_after_isotonic), "realisation_gap": float(t3.realisation_gap),
    "pass": bool(float(t3.ece_after_isotonic) < 0.10 and abs(float(t3.realisation_gap)) <= 0.03)},
 "G6_reliability_layer_lt_100ms": {"value_ms": rl, "pass": bool(rl < 100)},
}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
freeze = {
 "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
 "seed": cfg.seed, "python": platform.python_version(),
 "config_sha256_16": sha(ROOT / "configs/default.yaml"),
 "data_sha": {p.name: sha(p) for p in sorted((ROOT / "data/raw").glob("*.json"))},
 "gates": gates,
 "headline": {
   "test_questions": int(t1.loc["test", "questions"]),
   "test_articles": int(t1.loc["test", "articles"]),
   "base_accuracy": acc,
   "P1_error_auc": float(t2.loc["P1", "error_auc"]),
   "P1_error_auc_ci": [float(t2.loc["P1", "error_auc_lo"]), float(t2.loc["P1", "error_auc_hi"])],
   "B5_error_auc": float(t2.loc["B5", "error_auc"]),
   "B4_error_auc": float(t2.loc["B4", "error_auc"]),
   "delta_aurc_P1_B5": e2["delta_aurc_P1_minus_B5"], "ci": [e2["lo"], e2["hi"]],
   "ece_after": float(t3.ece_after_isotonic),
   "median_e2e_ms": float(t7.loc["end-to-end query (retrieval+reader+features)", "median_ms"]),
   "reliability_layer_ms": rl,
   "transfer_A_to_B_P1_auc": float(t6[(t6.method=="P1")&(t6.setting=="A -> B (transfer)")]["error_auc"].iloc[0]),
 },
 "tables": {p.name: sha(p) for p in sorted((ROOT / "results/tables").glob("*.csv"))},
}
json.dump(freeze, open(ROOT / "results/RESULT_FREEZE.json", "w"), indent=2)
print("\n--- GATES ---")
for k, v in gates.items(): print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
print("\nfrozen ->", ROOT / "results/RESULT_FREEZE.json")
