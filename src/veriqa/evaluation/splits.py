"""Article-level splitting. THE ONLY PLACE SPLITS ARE CREATED (D-004).

Four partitions are required because two models are trained: the reader must
not be evaluated on its own training articles, and the reliability model must
see a realistic reader error distribution, which it cannot if it is fitted on
articles the reader has already memorised.
"""
from __future__ import annotations
import random

PARTS = ("reader_train", "rel_train", "rel_calib", "test")


def split_articles(articles, fractions: dict, seed: int) -> dict:
    """Partition article names into four disjoint sets.

    Deterministic given (articles, seed). Returns {part: set[str]}.
    """
    arts = sorted(set(articles))
    if len(arts) < len(PARTS):
        raise ValueError(f"need >= {len(PARTS)} articles, got {len(arts)}")
    rng = random.Random(seed)
    rng.shuffle(arts)

    n = len(arts)
    counts, assigned = {}, 0
    for part in PARTS[:-1]:
        counts[part] = int(round(fractions[part] * n))
        assigned += counts[part]
    counts[PARTS[-1]] = n - assigned
    if counts[PARTS[-1]] < 1:
        raise ValueError("test partition would be empty; add more articles")

    out, i = {}, 0
    for part in PARTS:
        out[part] = set(arts[i:i + counts[part]])
        i += counts[part]

    # Invariant: pairwise disjoint and exhaustive.
    seen = set()
    for part in PARTS:
        if seen & out[part]:
            raise AssertionError(f"article leak into {part}")
        seen |= out[part]
    assert seen == set(arts), "articles lost during splitting"
    return out


def assign_questions(questions, article_split: dict) -> dict:
    """Map each partition to the list of question indices belonging to it."""
    lookup = {a: part for part, arts in article_split.items() for a in arts}
    out = {p: [] for p in PARTS}
    for i, q in enumerate(questions):
        out[lookup[q.article]].append(i)
    return out
