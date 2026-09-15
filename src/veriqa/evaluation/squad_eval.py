"""Official SQuAD normalisation and EM / F1, reimplemented from the published rules."""
from __future__ import annotations
import re, string
from collections import Counter

_ART = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNC = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNC)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1(pred: str, golds) -> float:
    best = 0.0
    pt = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(pt), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_correct(pred: str | None, golds, is_impossible: bool, thr: float = 0.5) -> bool:
    """Correctness label (D-005).

    Unanswerable: correct iff the system returned nothing.
    Answerable   : correct iff it returned something scoring F1 >= thr.
    """
    abstained = pred is None or not str(pred).strip()
    if is_impossible:
        return abstained
    return (not abstained) and f1(pred, golds) >= thr
