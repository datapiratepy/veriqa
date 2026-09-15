"""Hybrid retrieval: LSA dense similarity + BM25 lexical, with sibling masking.

No pretrained neural weights are used. The dense representation is truncated
SVD over a TF-IDF matrix (latent semantic analysis), which is a classical
embedding method that runs on CPU and requires no model download. See
docs/implementation/READER_BACKEND.md for why this matters.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

TOKEN = r"(?u)\b\w\w+\b"


def tokenize(text: str) -> list:
    import re
    return re.findall(r"\w+", text.lower())


@dataclass
class Retrieved:
    pids: np.ndarray        # (k,) passage ids, best first
    dense: np.ndarray       # (k,) dense similarity
    bm25: np.ndarray        # (k,) bm25 score (min-max normalised over candidates)
    hybrid: np.ndarray      # (k,) fused score
    all_hybrid: np.ndarray  # (k_wide,) top scores for distribution features


class HybridRetriever:
    def __init__(self, svd_dim: int = 256, alpha: float = 0.5, seed: int = 0):
        self.svd_dim, self.alpha, self.seed = svd_dim, alpha, seed

    def fit(self, passages):
        """passages: list[Passage]. Builds TF-IDF -> SVD dense index and BM25."""
        self.pids = np.array([p.pid for p in passages])
        self.article = np.array([p.article for p in passages])
        texts = [p.text for p in passages]

        self.tfidf = TfidfVectorizer(lowercase=True, token_pattern=TOKEN,
                                     sublinear_tf=True, min_df=2)
        X = self.tfidf.fit_transform(texts)
        dim = min(self.svd_dim, min(X.shape) - 1)
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed)
        self.P = normalize(self.svd.fit_transform(X)).astype(np.float32)

        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self._pid_row = {int(p): i for i, p in enumerate(self.pids)}
        return self

    def _mask(self, scores, article, gold_pid):
        """Protocol B: drop sibling paragraphs of the question's own article."""
        same = self.article == article
        same[self._pid_row[int(gold_pid)]] = False
        scores = scores.copy()
        scores[same] = -np.inf
        return scores

    def search(self, question: str, article: str, gold_pid: int, k: int,
               k_wide: int = 20) -> Retrieved:
        q = normalize(self.svd.transform(self.tfidf.transform([question])))
        dense = (self.P @ q[0].astype(np.float32))
        lex = np.asarray(self.bm25.get_scores(tokenize(question)), dtype=np.float32)

        dense_m = self._mask(dense, article, gold_pid)
        lex_m = self._mask(lex, article, gold_pid)

        valid = np.isfinite(dense_m)
        d = np.where(valid, dense_m, 0.0)
        l = np.where(valid, lex_m, 0.0)
        l = (l - l[valid].min()) / (np.ptp(l[valid]) + 1e-9)
        d01 = (d - d[valid].min()) / (np.ptp(d[valid]) + 1e-9)
        fused = np.where(valid, self.alpha * d01 + (1 - self.alpha) * l, -np.inf)

        wide = np.argsort(-fused)[:k_wide]
        top = wide[:k]
        return Retrieved(pids=self.pids[top], dense=dense[top], bm25=l[top],
                         hybrid=fused[top], all_hybrid=fused[wide])
