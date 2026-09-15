"""Shared query service used by both the Streamlit app and the FastAPI backend."""
from __future__ import annotations
import json, pickle, sqlite3, time
from pathlib import Path
import numpy as np
from .config import load_config, ROOT
from .ingestion.squad import load_corpus, Passage
from .reliability.features import extract_all, ALL, to_matrix
from .reader.spans import PassageIndex

DB = ROOT / "data" / "veriqa.sqlite"


def _db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS queries(
        ts REAL, question TEXT, answer TEXT, risk REAL, abstained INT,
        reason TEXT, passage_id INT)""")
    return con


class VeriQAService:
    """Loads frozen artefacts once and answers questions with an abstention gate."""

    def __init__(self, which="A"):
        self.cfg = load_config()
        self.corpus = load_corpus(ROOT / f"data/processed/corpus_{which}.json")
        art = pickle.load(open(ROOT / f"models/artifacts_{which}.pkl", "rb"))
        self.retr, self.reader = art["retriever"], art["reader"]
        rel = pickle.load(open(ROOT / "models/reliability_A.pkl", "rb"))
        self.risk_model, self.default_threshold = rel["m_p1"], rel["threshold"]
        self.by_pid = {p.pid: p for p in self.corpus.passages}
        self._pi = {}

    def ask(self, question: str, threshold: float | None = None, k: int | None = None):
        thr = self.default_threshold if threshold is None else threshold
        k = k or self.cfg.retrieval.k
        t0 = time.perf_counter()
        # A user query has no gold article; use a sentinel so nothing is masked.
        ret = self.retr.search(question, article="__user__", gold_pid=self.corpus.passages[0].pid, k=k)
        t1 = time.perf_counter()
        passages = [self.by_pid[int(p)] for p in ret.pids]
        out = self.reader.read(question, passages, ret.hybrid, self._pi)
        t2 = time.perf_counter()
        feats = extract_all(ret, out, question, passages, self.cfg.retrieval.sim_tau)
        risk = float(self.risk_model.risk(to_matrix([feats], ALL))[0])
        t3 = time.perf_counter()
        abstain = risk > thr
        reasons = []
        if feats["r_sim_top1"] < 0.5: reasons.append("weak evidence: low top-1 retrieval score")
        if feats["a_answer_in_top2"] == 0: reasons.append("no corroboration in the second passage")
        if feats["a_majority_frac"] < 0.5: reasons.append("retrieved passages disagree")
        if feats["d_span_prob"] < 0.1: reasons.append("reader is unsure of the span")
        span_txt = out.answer
        src = next((p for p in passages if p.pid == out.passage_pid), passages[0] if passages else None)
        res = {"question": question, "answer": None if abstain else span_txt,
               "candidate_answer": span_txt, "risk": risk, "abstained": bool(abstain),
               "threshold": thr,
               "reason": "; ".join(reasons) if abstain else "",
               "passage": src.text if src else "", "passage_id": int(out.passage_pid),
               "article": src.article if src else "",
               "evidence": [{"pid": int(p.pid), "article": p.article,
                             "score": float(s), "text": p.text[:400]}
                            for p, s in zip(passages, ret.hybrid)],
               "features": feats,
               "timing_ms": {"retrieval": (t1-t0)*1e3, "reader": (t2-t1)*1e3,
                             "reliability": (t3-t2)*1e3, "total": (t3-t0)*1e3}}
        try:
            con = _db(); con.execute("INSERT INTO queries VALUES (?,?,?,?,?,?,?)",
                (time.time(), question, span_txt, risk, int(abstain), res["reason"],
                 int(out.passage_pid))); con.commit(); con.close()
        except Exception:
            pass
        return res
