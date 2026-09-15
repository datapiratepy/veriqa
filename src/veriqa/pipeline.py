"""End-to-end pipeline: corpus -> retrieval -> reader -> reliability features."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np

from .config import load_config
from .ingestion.squad import load_squad, build_corpus
from .retrieval.hybrid import HybridRetriever, tokenize
from .reader.extractive import ExtractiveReader
from .reader.spans import PassageIndex
from .reliability.features import extract_all
from .evaluation.splits import split_articles, assign_questions
from .evaluation.squad_eval import is_correct


def build_idf(passages):
    from collections import Counter
    df, N = Counter(), len(passages)
    for p in passages:
        df.update(set(tokenize(p.text)))
    return {w: math.log(N / (1 + c)) + 1.0 for w, c in df.items()}


def prepare(cfg, n_articles=None, max_q=None, which="A", seed=None):
    seed = cfg.seed if seed is None else seed
    src = cfg.data.train_json if which == "A" else cfg.data.dev_json
    raw = load_squad(Path(__file__).resolve().parents[2] / src)
    corpus = build_corpus(raw, which,
                          n_articles=n_articles if n_articles is not None else (
                              cfg.corpus.n_articles_A if which == "A" else None),
                          max_q_per_article=max_q if max_q is not None
                          else cfg.corpus.max_questions_per_article,
                          seed=seed)
    art_split = split_articles(corpus.articles, cfg.splits, seed)
    qidx = assign_questions(corpus.questions, art_split)
    return corpus, art_split, qidx


def fit_reader(cfg, corpus, qidx, idf, verbose=True):
    by_pid = {p.pid: p for p in corpus.passages}
    train_qs = [corpus.questions[i] for i in qidx["reader_train"]]
    if verbose:
        print(f"  fitting reader on {len(train_qs):,} questions "
              f"from {len({q.article for q in train_qs})} articles")
    rd = ExtractiveReader(max_span=cfg.reader.max_span_tokens,
                          neg_per_pos=cfg.reader.neg_per_pos, seed=cfg.seed)
    rd.fit(train_qs, by_pid, idf, verbose=verbose)
    return rd


def run_questions(cfg, corpus, retr, reader, indices, out_path=None,
                  verbose=True, log_every=500, budget_s=None):
    """Retrieve + read + featurise. Resumable: completed qids are skipped."""
    by_pid = {p.pid: p for p in corpus.passages}
    done, rows = set(), []
    if out_path and Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # tolerate a truncated final line after a kill
                rows.append(r); done.add(r["qid"])
        if verbose and done:
            print(f"  resuming: {len(done):,} already complete")
    fh = open(out_path, "a") if out_path else None
    pi_cache, t0, timings = {}, time.time(), []
    todo = [i for i in indices if corpus.questions[i].qid not in done]
    for n, i in enumerate(todo, 1):
        q = corpus.questions[i]
        t_a = time.perf_counter()
        ret = retr.search(q.question, q.article, q.gold_pid, cfg.retrieval.k)
        t_b = time.perf_counter()
        passages = [by_pid[int(p)] for p in ret.pids]
        out = reader.read(q.question, passages, ret.hybrid, pi_cache)
        t_c = time.perf_counter()
        feats = extract_all(ret, out, q.question, passages, cfg.retrieval.sim_tau)
        t_d = time.perf_counter()
        rec = {"qid": q.qid, "article": q.article, "is_impossible": q.is_impossible,
               "answers": q.answers, "pred": out.answer, "gold_pid": q.gold_pid,
               "retrieved": [int(x) for x in ret.pids],
               "gold_retrieved": bool(q.gold_pid in set(int(x) for x in ret.pids)),
               "features": feats,
               "t_retrieval": t_b - t_a, "t_reader": t_c - t_b, "t_features": t_d - t_c}
        rows.append(rec)
        timings.append(t_d - t_a)
        if fh:
            fh.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                fh.flush()
        if budget_s is not None and (time.time() - t0) > budget_s:
            if verbose:
                print(f"    budget reached after {n:,} questions; resumable")
            break
        if verbose and n % log_every == 0:
            el = time.time() - t0
            print(f"    {n:,}/{len(todo):,}  {el:.0f}s  eta {el/n*(len(todo)-n):.0f}s")
    if fh:
        fh.close()
    return rows


def label_rows(rows, thr=0.5):
    y = np.array([0.0 if is_correct(r["pred"], r["answers"], r["is_impossible"], thr)
                  else 1.0 for r in rows])
    return y
