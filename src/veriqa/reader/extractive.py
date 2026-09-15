"""Classical feature-based extractive reader with an explicit null-answer head.

Two models are fitted on the reader_train articles only:
  * a span scorer  -> P(this candidate span is the answer)
  * a null head    -> P(this question has no answer in this passage)

This mirrors the structure of a transformer SQuAD 2.0 reader (best-span score
plus a no-answer score) without requiring pretrained weights. See
docs/implementation/READER_BACKEND.md.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingClassifier

from ..evaluation.squad_eval import normalize_answer
from .spans import (PassageIndex, span_features, span_text, qtype,
                    candidate_mask, N_FEATS)

NULL_FEATS = 8


@dataclass
class ReadOut:
    answer: str
    span_prob: float          # softmax prob of the chosen span in its passage
    null_score: float         # P(no answer) from the null head, best passage
    span_len: int
    span_entropy: float       # entropy of the span softmax
    passage_pid: int
    per_passage: list         # [(pid, answer, span_prob, null_score)]


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class ExtractiveReader:
    def __init__(self, max_span=8, neg_per_pos=30, seed=0, top_sentences=3):
        self.max_span, self.neg_per_pos, self.seed = max_span, neg_per_pos, seed
        self.top_sentences = top_sentences
        self.span_model = None
        self.null_model = None
        self.idf = {}

    # ---------------------------------------------------------------- fitting
    def _content(self, question):
        """Content words, tokenised exactly as passages are.

        str.split() left trailing punctuation attached, so the final and often
        most informative content word of every question never matched a passage
        token. Regex tokenisation fixes it. See D-006.
        """
        import re
        from .spans import STOP
        return {w for w in re.findall(r"\w+", question.lower())
                if w not in STOP and len(w) > 1}

    def _null_features(self, X_span, scores, q_content, pi):
        top = np.sort(scores)[::-1][:5]
        top = np.pad(top, (0, max(0, 5 - len(top))), constant_values=0.0)
        ov = 0.0
        if q_content:
            ov = len(set(pi.low) & q_content) / len(q_content)
        return np.array([top[0], top[:3].mean(), top.mean(), top[0] - top[1] if len(scores) > 1 else 0.0,
                         float(scores.std()) if len(scores) else 0.0, ov,
                         float(len(q_content)), float(pi.n)], dtype=np.float32)

    def fit(self, questions, passage_by_pid, idf_map, verbose=True):
        rng = np.random.default_rng(self.seed)
        self.idf = idf_map
        Xs, ys = [], []
        cache = {}
        for qi, q in enumerate(questions):
            p = passage_by_pid[q.gold_pid]
            pi = cache.get(q.gold_pid)
            if pi is None:
                pi = cache[q.gold_pid] = PassageIndex(p.text, self.max_span)
            if pi.n == 0:
                continue
            qc = self._content(q.question)
            X = span_features(pi, qc, idf_map, qtype(q.question), 1.0)
            keep = candidate_mask(pi, qc, self.top_sentences)
            X, kidx = X[keep], np.where(keep)[0]
            if len(X) == 0:
                continue
            if q.is_impossible or not q.answers:
                idx = rng.choice(len(X), size=min(len(X), self.neg_per_pos), replace=False)
                Xs.append(X[idx]); ys.append(np.zeros(len(idx)))
                continue
            golds = {normalize_answer(a) for a in q.answers}
            texts = np.array([normalize_answer(span_text(pi, int(i))) for i in kidx])
            pos = np.where(np.isin(texts, list(golds)))[0]
            if len(pos) == 0:
                continue
            neg_pool = np.setdiff1d(np.arange(len(X)), pos)
            neg = rng.choice(neg_pool, size=min(len(neg_pool), self.neg_per_pos * len(pos)),
                             replace=False)
            Xs.append(np.vstack([X[pos], X[neg]]))
            ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        X, y = np.vstack(Xs), np.concatenate(ys)
        if verbose:
            print(f"  span scorer training rows={len(y):,} positives={int(y.sum()):,}")
        self.span_model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=self.seed,
            class_weight="balanced").fit(X, y)

        # --- null head, fitted on the same articles, gold passage only -------
        Xn, yn = [], []
        for q in questions:
            p = passage_by_pid[q.gold_pid]
            pi = cache.get(q.gold_pid) or PassageIndex(p.text, self.max_span)
            if pi.n == 0:
                continue
            qc = self._content(q.question)
            Xf = span_features(pi, qc, idf_map, qtype(q.question), 1.0)
            keep = candidate_mask(pi, qc, self.top_sentences)
            Xf = Xf[keep] if keep.any() else Xf
            s = self.span_model.predict_proba(Xf)[:, 1]
            Xn.append(self._null_features(Xf, s, qc, pi))
            yn.append(float(q.is_impossible))
        Xn, yn = np.vstack(Xn), np.array(yn)
        if verbose:
            print(f"  null head training rows={len(yn):,} impossible={int(yn.sum()):,}")
        self.null_model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=self.seed).fit(Xn, yn)
        return self

    # -------------------------------------------------------------- inference
    def read(self, question: str, passages, scores, pi_cache: dict) -> ReadOut:
        """passages: list[Passage] top-k. scores: retrieval hybrid scores."""
        qc = self._content(question)
        qt = qtype(question)
        per, best = [], None
        for p, sc in zip(passages, scores):
            pi = pi_cache.get(p.pid)
            if pi is None:
                pi = pi_cache[p.pid] = PassageIndex(p.text, self.max_span)
            if pi.n == 0:
                continue
            X = span_features(pi, qc, self.idf, qt, float(sc))
            keep = candidate_mask(pi, qc, self.top_sentences)
            kidx = np.where(keep)[0] if keep.any() else np.arange(len(X))
            X = X[kidx]
            if len(X) == 0:
                continue
            s = self.span_model.predict_proba(X)[:, 1]
            j_local = int(np.argmax(s))
            j = int(kidx[j_local])
            sm = _softmax(s * 10.0)
            ent = float(-(sm * np.log(sm + 1e-12)).sum())
            nul = float(self.null_model.predict_proba(
                self._null_features(X, s, qc, pi).reshape(1, -1))[:, 1][0])
            rec = dict(pid=p.pid, answer=span_text(pi, j), span_prob=float(sm[j_local]),
                       raw=float(s[j_local]), null=nul, ent=ent, length=int(pi.lens[j]))
            per.append(rec)
            if best is None or rec["raw"] > best["raw"]:
                best = rec
        if best is None:
            return ReadOut("", 0.0, 1.0, 0, 0.0, -1, [])
        return ReadOut(best["answer"], best["span_prob"], best["null"], best["length"],
                       best["ent"], best["pid"],
                       [(r["pid"], r["answer"], r["span_prob"], r["null"]) for r in per])
