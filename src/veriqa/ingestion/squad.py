"""SQuAD 2.0 loading and corpus construction under protocol B (D-002).

Corpus unit is the paragraph (D-003). Retrieval space for a question excludes
sibling paragraphs of its own article; see docs/research/SQUAD_ANSWERABILITY_PROTOCOL.md
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Passage:
    pid: int
    article: str
    para_idx: int
    text: str


@dataclass
class Question:
    qid: str
    article: str
    gold_pid: int
    question: str
    answers: list          # list[str]; empty when unanswerable
    is_impossible: bool


@dataclass
class Corpus:
    name: str
    passages: list = field(default_factory=list)
    questions: list = field(default_factory=list)

    @property
    def articles(self):
        return sorted({p.article for p in self.passages})

    def pid_article(self):
        return {p.pid: p.article for p in self.passages}

    def stats(self):
        n_imp = sum(q.is_impossible for q in self.questions)
        return {
            "name": self.name,
            "articles": len(self.articles),
            "passages": len(self.passages),
            "questions": len(self.questions),
            "unanswerable": n_imp,
            "unanswerable_frac": round(n_imp / max(len(self.questions), 1), 4),
        }


def load_squad(path: str | Path) -> list:
    with open(path) as fh:
        return json.load(fh)["data"]


def build_corpus(raw_articles: list, name: str, *, n_articles: int | None = None,
                 max_q_per_article: int | None = None, seed: int = 0) -> Corpus:
    """Build a Corpus from raw SQuAD article records.

    Sampling is article-level and seeded, so the corpus is reproducible.
    Questions are subsampled *within* an article to bound runtime without
    removing any article from the retrieval corpus.
    """
    rng = random.Random(seed)
    arts = sorted(raw_articles, key=lambda a: a["title"])
    if n_articles is not None and n_articles < len(arts):
        arts = rng.sample(arts, n_articles)
        arts.sort(key=lambda a: a["title"])

    corpus, pid = Corpus(name=name), 0
    for art in arts:
        title = art["title"]
        art_questions = []
        for pi, para in enumerate(art["paragraphs"]):
            corpus.passages.append(Passage(pid, title, pi, para["context"]))
            for qa in para["qas"]:
                answers = sorted({a["text"] for a in qa.get("answers", [])})
                art_questions.append(Question(
                    qid=qa["id"], article=title, gold_pid=pid,
                    question=qa["question"], answers=answers,
                    is_impossible=bool(qa.get("is_impossible", False)),
                ))
            pid += 1
        if max_q_per_article is not None and len(art_questions) > max_q_per_article:
            # Stratify the subsample so the answerable/unanswerable balance of
            # the article is preserved rather than drifting with the draw.
            imp = [q for q in art_questions if q.is_impossible]
            pos = [q for q in art_questions if not q.is_impossible]
            frac = len(imp) / len(art_questions)
            n_imp = min(len(imp), int(round(max_q_per_article * frac)))
            n_pos = min(len(pos), max_q_per_article - n_imp)
            art_questions = rng.sample(imp, n_imp) + rng.sample(pos, n_pos)
        art_questions.sort(key=lambda q: q.qid)
        corpus.questions.extend(art_questions)
    return corpus


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"name": corpus.name,
                   "passages": [asdict(p) for p in corpus.passages],
                   "questions": [asdict(q) for q in corpus.questions]}, fh)


def load_corpus(path: str | Path) -> Corpus:
    with open(path) as fh:
        d = json.load(fh)
    return Corpus(name=d["name"],
                  passages=[Passage(**p) for p in d["passages"]],
                  questions=[Question(**q) for q in d["questions"]])
