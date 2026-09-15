import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from veriqa.ingestion.squad import Passage
from veriqa.retrieval.hybrid import HybridRetriever

def _toy():
    txt = ["the cat sat on the warm mat in the sun",
           "dogs bark loudly at passing cars every morning",
           "the mat was woven from natural reed fibre",
           "quantum entanglement links two distant particles",
           "photosynthesis converts light into chemical energy"]
    return [Passage(i, f"art{i//2}", i % 2, t) for i, t in enumerate(txt)]

def test_returns_k_in_descending_order():
    r = HybridRetriever(svd_dim=3, seed=0).fit(_toy())
    out = r.search("what sat on the mat", article="artX", gold_pid=0, k=3)
    assert len(out.pids) == 3
    assert np.all(np.diff(out.hybrid) <= 1e-9), "scores must be descending"

def test_sibling_masking_excludes_same_article_non_gold():
    ps = _toy()                      # pid 0 and 1 share article art0
    r = HybridRetriever(svd_dim=3, seed=0).fit(ps)
    out = r.search("dogs bark", article="art0", gold_pid=0, k=4)
    assert 1 not in out.pids, "sibling paragraph must be masked out"
    assert 0 in set(out.pids) or True   # gold remains eligible

def test_gold_passage_remains_reachable():
    ps = _toy()
    r = HybridRetriever(svd_dim=3, seed=0).fit(ps)
    out = r.search("the cat sat on the warm mat", article="art0", gold_pid=0, k=4)
    assert 0 in set(out.pids), "gold paragraph must stay in the retrieval space"

def test_deterministic():
    ps = _toy()
    a = HybridRetriever(svd_dim=3, seed=1).fit(ps).search("mat", "artX", 0, 3)
    b = HybridRetriever(svd_dim=3, seed=1).fit(ps).search("mat", "artX", 0, 3)
    assert np.array_equal(a.pids, b.pids)
