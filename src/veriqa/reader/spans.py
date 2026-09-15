"""Candidate span generation and vectorised span features.

All features for every candidate span in a passage are produced with cumulative
sums and broadcasting, so a 107-token paragraph with max span length 8 yields
~850 candidates scored in a single numpy pass rather than a Python loop.
"""
from __future__ import annotations
import re
import numpy as np

WORD = re.compile(r"\w+|[^\w\s]")
STOP = set("""a an the of in on at to for with by from as is are was were be been being
and or but if then than that this these those it its he she they them his her their we you i
""".split())
SENT_END = {".", "!", "?"}

QTYPES = ("who", "when", "where", "what", "which", "why", "how_many", "how", "other")
N_FEATS = 20
PUNC = set(".,;:!?()[]\"'`-")
CTX = 4        # context window, tokens either side of a candidate span


def qtype(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("how many") or q.startswith("how much") or " how many " in q:
        return "how_many"
    for t in ("who", "when", "where", "which", "why", "what", "how"):
        if q.startswith(t) or f" {t} " in q[:40]:
            return t
    return "other"


def tokenize_offsets(text: str):
    """Return (tokens, char_spans) preserving the original surface text."""
    toks, spans = [], []
    for m in WORD.finditer(text):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans


class PassageIndex:
    """Pre-computed per-token arrays for one passage. Built once, reused per question."""

    __slots__ = ("text", "toks", "spans", "low", "n", "is_cap", "is_num",
                 "is_stop", "sent_id", "starts", "lens", "valid_mask")

    def __init__(self, text: str, max_span: int):
        self.text = text
        self.toks, self.spans = tokenize_offsets(text)
        self.low = np.array([t.lower() for t in self.toks], dtype=object)
        self.n = len(self.toks)
        self.is_cap = np.array([t[:1].isupper() for t in self.toks], dtype=np.float32)
        self.is_num = np.array([bool(re.fullmatch(r"[\d,.]+", t)) for t in self.toks],
                               dtype=np.float32)
        self.is_stop = np.array([t.lower() in STOP for t in self.toks], dtype=np.float32)
        sid, cur = np.zeros(self.n, dtype=np.int32), 0
        for i, t in enumerate(self.toks):
            sid[i] = cur
            if t in SENT_END:
                cur += 1
        self.sent_id = sid
        s = np.arange(self.n)[:, None]
        l = np.arange(1, max_span + 1)[None, :]
        m = (s + l) <= self.n
        st = np.broadcast_to(s, m.shape)[m].astype(np.int32)
        ln = np.broadcast_to(l, m.shape)[m].astype(np.int32)
        # Candidate filter: a SQuAD answer span essentially never begins or ends
        # on a stopword or a punctuation token. Dropping those keeps 90.4% of
        # gold answers (vs 92.7% unfiltered) while removing 64% of candidates,
        # which is what makes argmax over the candidate set tractable. See D-007.
        bad = np.array([(t.lower() in STOP) or (t in PUNC) for t in self.toks])
        keep = ~bad[st] & ~bad[st + ln - 1]
        self.starts, self.lens = st[keep], ln[keep]


def _cs(a):
    return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])


def span_features(pi: PassageIndex, q_content: set, idf_map: dict,
                  qt: str, passage_score: float) -> np.ndarray:
    """(n_candidates, N_FEATS) float32 feature matrix for every candidate span."""
    n = pi.n
    if n == 0:
        return np.zeros((0, N_FEATS), dtype=np.float32)

    in_q = np.array([w in q_content for w in pi.low], dtype=np.float32)
    idf = np.array([idf_map.get(w, 8.0) for w in pi.low], dtype=np.float32)

    # distance from each position to the nearest question-word occurrence
    pos = np.where(in_q > 0)[0]
    if len(pos):
        d = np.abs(np.arange(n)[:, None] - pos[None, :]).min(1).astype(np.float32)
    else:
        d = np.full(n, float(n), dtype=np.float32)

    # sentence-level question overlap
    n_sent = int(pi.sent_id.max()) + 1
    sent_ov = np.zeros(n_sent, dtype=np.float32)
    if q_content:
        for s in range(n_sent):
            toks = set(pi.low[pi.sent_id == s])
            sent_ov[s] = len(toks & q_content) / len(q_content)

    cs_cap, cs_num, cs_stop = _cs(pi.is_cap), _cs(pi.is_num), _cs(pi.is_stop)
    cs_inq, cs_idf = _cs(in_q), _cs(idf)

    st, ln = pi.starts, pi.lens
    en = st + ln
    L = ln.astype(np.float32)

    f_cap = ((cs_cap[en] - cs_cap[st]) / L).astype(np.float32)
    f_num = ((cs_num[en] - cs_num[st]) / L).astype(np.float32)
    f_stop = ((cs_stop[en] - cs_stop[st]) / L).astype(np.float32)
    f_inq = ((cs_inq[en] - cs_inq[st]) / L).astype(np.float32)
    f_idf = ((cs_idf[en] - cs_idf[st]) / L).astype(np.float32)
    f_dist = d[st]
    f_ov = sent_ov[pi.sent_id[st]]
    f_rel = (st / max(n, 1)).astype(np.float32)
    f_same = (pi.sent_id[st] == pi.sent_id[en - 1]).astype(np.float32)
    f_bnd = (pi.is_cap[st] > 0).astype(np.float32)

    want_cap = float(qt in ("who", "where", "which"))
    want_num = float(qt in ("when", "how_many"))

    # --- context features: what surrounds the span, not just the span itself.
    # A correct answer is usually adjacent to the question's own wording.
    cs_inq_ctx = cs_inq
    lo = np.maximum(st - CTX, 0)
    hi = np.minimum(en + CTX, n)
    f_lctx = ((cs_inq_ctx[st] - cs_inq_ctx[lo]) /
              np.maximum(st - lo, 1)).astype(np.float32)
    f_rctx = ((cs_inq_ctx[hi] - cs_inq_ctx[en]) /
              np.maximum(hi - en, 1)).astype(np.float32)
    f_proper = (f_cap == 1.0).astype(np.float32)
    # rank of the span's sentence among sentences by question overlap (0 = best)
    sent_rank = np.empty(n_sent, dtype=np.float32)
    sent_rank[np.argsort(-sent_ov, kind="mergesort")] = np.arange(n_sent)
    f_srank = sent_rank[pi.sent_id[st]]

    return np.column_stack([
        L, f_cap, f_num, f_stop, f_inq, f_idf, f_dist, f_ov, f_rel, f_same, f_bnd,
        f_cap * want_cap, f_num * want_num,
        np.full(len(st), passage_score, dtype=np.float32),
        np.full(len(st), float(len(q_content)), dtype=np.float32),
        f_ov * (1.0 - f_inq),
        f_lctx, f_rctx, f_proper, f_srank,
    ]).astype(np.float32)


def sentence_overlap(pi: PassageIndex, q_content: set) -> np.ndarray:
    """Question-content-word overlap of each sentence in the passage."""
    n_sent = int(pi.sent_id.max()) + 1 if pi.n else 0
    ov = np.zeros(max(n_sent, 1), dtype=np.float32)
    if q_content and pi.n:
        for s in range(n_sent):
            toks = set(pi.low[pi.sent_id == s])
            ov[s] = len(toks & q_content) / len(q_content)
    return ov


def candidate_mask(pi: PassageIndex, q_content: set, top_sentences: int = 3):
    """Restrict candidates to spans inside the best-matching sentences.

    Stage 1 of a two-stage reader. 68.2% of SQuAD gold answers lie in the single
    highest-overlap sentence and 83.3% within the top three, so this discards
    most of the candidate space at a small recall cost and lets the span scorer
    solve a far easier discrimination problem. See D-008.
    """
    if pi.n == 0:
        return np.zeros(0, dtype=bool)
    ov = sentence_overlap(pi, q_content)
    keep = set(np.argsort(-ov, kind="mergesort")[:top_sentences].tolist())
    return np.array([int(sid) in keep for sid in pi.sent_id[pi.starts]], dtype=bool)


def span_text(pi: PassageIndex, i: int) -> str:
    a = pi.spans[int(pi.starts[i])][0]
    b = pi.spans[int(pi.starts[i] + pi.lens[i] - 1)][1]
    return pi.text[a:b]
