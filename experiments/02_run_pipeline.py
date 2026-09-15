"""E-prep: run retrieval + reader + feature extraction over all questions.

Resumable: re-running continues from the last written line.
"""
import sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa import pipeline as P
from veriqa.ingestion.squad import load_corpus

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    cfg = load_config(); set_all_seeds(cfg.seed)
    corpus = load_corpus(ROOT / f"data/processed/corpus_{which}.json")
    art = pickle.load(open(ROOT / f"models/artifacts_{which}.pkl", "rb"))
    qidx = art["qidx"]
    # every partition except reader_train is needed downstream
    parts = ["rel_train", "rel_calib", "test"] if which == "A" else ["rel_train", "rel_calib", "test"]
    idx = sorted(i for p in parts for i in qidx[p])
    print(f"[{which}] running {len(idx):,} questions over {len(corpus.passages):,} passages")
    t = time.time()
    rows = P.run_questions(cfg, corpus, art["retriever"], art["reader"], idx,
                           out_path=ROOT / f"results/raw/predictions_{which}.jsonl",
                           verbose=True, log_every=1000,
                           budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(f"[{which}] {len(rows):,} rows in {time.time()-t:.0f}s")
