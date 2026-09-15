import sys, pathlib, pickle, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from veriqa.config import load_config
from veriqa.evaluation.splits import split_articles, assign_questions, PARTS
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_partitions_are_article_disjoint():
    cfg = load_config()
    arts = [f"art{i}" for i in range(120)]
    sp = split_articles(arts, cfg.splits, cfg.seed)
    seen = set()
    for p in PARTS:
        assert not (seen & sp[p]), f"{p} shares an article with an earlier partition"
        seen |= sp[p]
    assert seen == set(arts)

def test_split_is_deterministic():
    cfg = load_config(); arts = [f"art{i}" for i in range(60)]
    assert split_articles(arts, cfg.splits, cfg.seed) == split_articles(arts, cfg.splits, cfg.seed)

def test_saved_run_has_no_article_overlap():
    p = ROOT / "models/artifacts_A.pkl"
    if not p.exists(): return
    sp = pickle.load(open(p, "rb"))["art_split"]
    seen = set()
    for k, v in sp.items():
        assert not (seen & set(v)), f"article leak in {k}"
        seen |= set(v)

def test_no_question_appears_in_two_partitions():
    p = ROOT / "results/raw/predictions_A.jsonl"
    if not p.exists(): return
    sp = pickle.load(open(ROOT / "models/artifacts_A.pkl", "rb"))["art_split"]
    lookup = {a: k for k, v in sp.items() for a in v}
    seen = {}
    for line in open(p):
        r = json.loads(line)
        part = lookup[r["article"]]
        assert seen.setdefault(r["qid"], part) == part
