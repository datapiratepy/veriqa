"""E-prep: build Corpus A (main) and Corpus B (domain shift), fit index+reader."""
import sys, json, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from veriqa.config import load_config
from veriqa.utils.seeds import set_all_seeds
from veriqa import pipeline as P
from veriqa.retrieval.hybrid import HybridRetriever
from veriqa.ingestion.squad import save_corpus

def build(which, cfg):
    n = cfg.corpus.n_articles_A if which == "A" else None
    corpus, art_split, qidx = P.prepare(cfg, n_articles=n, which=which)
    print(f"[{which}] {corpus.stats()}")
    print(f"[{which}] split articles " + str({k: len(v) for k, v in art_split.items()}))
    print(f"[{which}] split questions " + str({k: len(v) for k, v in qidx.items()}))
    save_corpus(corpus, ROOT / f"data/processed/corpus_{which}.json")
    idf = P.build_idf(corpus.passages)
    t = time.time()
    retr = HybridRetriever(cfg.retrieval.svd_dim, cfg.retrieval.hybrid_alpha,
                           cfg.seed).fit(corpus.passages)
    t_index = time.time() - t
    print(f"[{which}] index built in {t_index:.1f}s over {len(corpus.passages)} passages")
    t = time.time()
    reader = P.fit_reader(cfg, corpus, qidx, idf)
    t_reader = time.time() - t
    print(f"[{which}] reader fitted in {t_reader:.1f}s")
    with open(ROOT / f"models/artifacts_{which}.pkl", "wb") as fh:
        pickle.dump({"retriever": retr, "reader": reader, "idf": idf,
                     "art_split": {k: sorted(v) for k, v in art_split.items()},
                     "qidx": qidx, "t_index": t_index, "t_reader": t_reader}, fh)
    return corpus.stats(), t_index, t_reader

if __name__ == "__main__":
    cfg = load_config(); set_all_seeds(cfg.seed)
    out = {}
    for which in ("A", "B"):
        s, ti, tr = build(which, cfg)
        out[which] = {"stats": s, "t_index_s": ti, "t_reader_fit_s": tr}
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results/raw/corpus_build.json", "w"), indent=2)
    print("done")
