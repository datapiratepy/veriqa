"""E7 latency and memory. Per-stage timings come from the actual pipeline run."""
import sys, json, pickle, resource, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.reliability.features import ALL, to_matrix
cfg = load_config()

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(ROOT / "results/raw/predictions_A.jsonl")]
    build = json.load(open(ROOT / "results/raw/corpus_build.json"))
    rel = pickle.load(open(ROOT / "models/reliability_A.pkl", "rb"))["m_p1"]
    X = to_matrix([r["features"] for r in rows[:2000]], ALL)

    # gate cost measured directly, 100 repeats over single queries
    gate = []
    for i in range(100):
        t = time.perf_counter(); rel.risk(X[i:i + 1]); gate.append(time.perf_counter() - t)
    gate = np.array(gate)

    def row(name, v):
        v = np.asarray(v) * 1000.0
        return {"stage": name, "median_ms": float(np.median(v)),
                "mean_ms": float(v.mean()), "p95_ms": float(np.percentile(v, 95)),
                "n": int(len(v))}

    t_r = [r["t_retrieval"] for r in rows]; t_d = [r["t_reader"] for r in rows]
    t_f = [r["t_features"] for r in rows]
    e2e = [a + b + c for a, b, c in zip(t_r, t_d, t_f)]
    recs = [row("retrieval", t_r), row("reader", t_d), row("reliability features", t_f),
            row("risk model + gate", gate),
            row("end-to-end query (retrieval+reader+features)", e2e)]
    rl = np.median(np.array(t_f)) * 1000 + float(np.median(gate)) * 1000
    recs.append({"stage": "RELIABILITY LAYER TOTAL (features + gate)",
                 "median_ms": rl, "mean_ms": float("nan"),
                 "p95_ms": float(np.percentile(np.array(t_f) * 1000, 95)
                                 + np.percentile(gate * 1000, 95)), "n": len(t_f)})
    T7 = pd.DataFrame(recs)
    T7.to_csv(ROOT / "results/tables/T7_latency.csv", index=False)
    print("--- E7 LATENCY (per query, CPU) ---"); print(T7.round(3).to_string(index=False))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {"peak_rss_mb_this_process": peak_mb,
            "index_build_s_A": build["A"]["t_index_s"],
            "reader_fit_s_A": build["A"]["t_reader_fit_s"],
            "passages_A": build["A"]["stats"]["passages"],
            "model_sizes_mb": {
                "artifacts_A.pkl": (ROOT / "models/artifacts_A.pkl").stat().st_size / 1e6,
                "reliability_A.pkl": (ROOT / "models/reliability_A.pkl").stat().st_size / 1e6},
            "note": "No pretrained neural weights are loaded; see READER_BACKEND.md"}
    json.dump(meta, open(ROOT / "results/raw/E7_resources.json", "w"), indent=2)
    print("\n" + json.dumps(meta, indent=2))
