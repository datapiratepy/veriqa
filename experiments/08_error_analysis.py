"""E8 error analysis + F1-threshold sensitivity + gate evaluation + figures."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa.evaluation.metrics import (aurc, error_auc, risk_coverage_curve, ece,
                                       paired_bootstrap_diff)
from veriqa.evaluation.squad_eval import is_correct, f1 as sf1
from veriqa.reliability.features import ALL, FAMILY, to_matrix
from veriqa.reliability.calibrator import RiskModel
from veriqa.reader.spans import qtype
cfg = load_config(); set_all_seeds(cfg.seed)

rows = [json.loads(l) for l in open(ROOT / "results/raw/predictions_A.jsonl")]
art = pickle.load(open(ROOT / "models/artifacts_A.pkl", "rb"))
part = {p: set(a) for p, a in art["art_split"].items()}
by = {p: [r for r in rows if r["article"] in part[p]] for p in part}
tr, cal, te = by["rel_train"], by["rel_calib"], by["test"]

def lab(rs, thr): return np.array([0.0 if is_correct(r["pred"], r["answers"],
                                   r["is_impossible"], thr) else 1.0 for r in rs])
def fit(cols, thr):
    m = RiskModel(max_iter=cfg.reliability.gbm_max_iter,
                  learning_rate=cfg.reliability.gbm_learning_rate, seed=cfg.seed)
    m.fit(to_matrix([r["features"] for r in tr], cols), lab(tr, thr))
    m.calibrate(to_matrix([r["features"] for r in cal], cols), lab(cal, thr))
    return m

# ---- F1 threshold sensitivity ------------------------------------------------
g_te = np.array([r["article"] for r in te]); sens = []
for thr in [cfg.evaluation.f1_threshold] + list(cfg.evaluation.f1_sensitivity):
    c = 1.0 - lab(te, thr)
    rp = fit(ALL, thr).risk(to_matrix([r["features"] for r in te], ALL))
    rb = fit(FAMILY["reader"], thr).risk(to_matrix([r["features"] for r in te], FAMILY["reader"]))
    d, lo, hi = paired_bootstrap_diff(rp, rb, c, g_te, aurc, 500, cfg.seed)
    sens.append({"f1_threshold": thr, "base_error_rate": float(1 - c.mean()),
                 "P1_auc": error_auc(rp, c), "B5_auc": error_auc(rb, c),
                 "P1_aurc": aurc(rp, c), "B5_aurc": aurc(rb, c),
                 "delta_aurc": d, "lo": lo, "hi": hi,
                 "P1_beats_B5": bool(hi < 0)})
T8 = pd.DataFrame(sens); T8.to_csv(ROOT / "results/tables/T8_f1_sensitivity.csv", index=False)
print("--- F1 THRESHOLD SENSITIVITY ---"); print(T8.round(4).to_string(index=False))

# ---- error taxonomy ----------------------------------------------------------
THR = cfg.evaluation.f1_threshold
c_te = 1.0 - lab(te, THR)
m_p1 = fit(ALL, THR); risk = m_p1.risk(to_matrix([r["features"] for r in te], ALL))
def classify(r):
    if r["is_impossible"]:
        return "unanswerable question answered anyway"
    if not r["gold_retrieved"]:
        return "retrieval failure (gold passage not in top-k)"
    f = sf1(r["pred"], r["answers"])
    if f == 0: return "wrong span (gold retrieved, no overlap)"
    if f < THR: return "partial span overlap below threshold"
    return "correct"
tax = pd.Series([classify(r) for r in te]).value_counts().rename_axis("category").reset_index(name="count")
tax["fraction"] = tax["count"] / len(te)
tax.to_csv(ROOT / "results/tables/T9_error_taxonomy.csv", index=False)
print("\n--- E8 ERROR TAXONOMY (test, n=%d) ---" % len(te)); print(tax.round(4).to_string(index=False))

# by question type
qt = pd.DataFrame({"qtype": [qtype(r["question"] if "question" in r else "") for r in te]}) \
    if False else None
rows_qt = []
for r, c, rk in zip(te, c_te, risk):
    rows_qt.append({"qtype": qtype(r.get("question", "")), "correct": c, "risk": rk,
                    "is_impossible": r["is_impossible"]})
# question text was not stored in predictions; recover from the corpus
from veriqa.ingestion.squad import load_corpus
corpus = load_corpus(ROOT / "data/processed/corpus_A.json")
qmap = {q.qid: q.question for q in corpus.questions}
for d, r in zip(rows_qt, te): d["qtype"] = qtype(qmap.get(r["qid"], ""))
QT = pd.DataFrame(rows_qt).groupby("qtype").agg(
    n=("correct", "size"), accuracy=("correct", "mean"), mean_risk=("risk", "mean")).reset_index()
QT = QT.sort_values("n", ascending=False)
QT.to_csv(ROOT / "results/tables/T10_by_question_type.csv", index=False)
print("\n--- ACCURACY BY QUESTION TYPE ---"); print(QT.round(4).to_string(index=False))

# three real verbatim examples
ex = []
order = np.argsort(risk)
for tag, i in (("lowest-risk answered", order[0]), ("highest-risk (correctly flagged)", order[-1])):
    r = te[int(i)]
    ex.append({"case": tag, "question": qmap.get(r["qid"], ""), "prediction": r["pred"],
               "gold": r["answers"], "is_impossible": r["is_impossible"],
               "risk": float(risk[int(i)]), "correct": bool(c_te[int(i)])})
mis = [j for j in order if c_te[j] == 1.0]
if len(mis):
    r = te[int(mis[0])]
    ex.append({"case": "lowest-risk correct answer", "question": qmap.get(r["qid"], ""),
               "prediction": r["pred"], "gold": r["answers"],
               "is_impossible": r["is_impossible"], "risk": float(risk[int(mis[0])]),
               "correct": True})
json.dump(ex, open(ROOT / "results/raw/E8_examples.json", "w"), indent=2)
print("\n--- REAL EXAMPLES ---")
for e in ex: print(" ", {k: (v[:70] if isinstance(v, str) else v) for k, v in e.items()})
