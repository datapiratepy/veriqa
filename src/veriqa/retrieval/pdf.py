
"""Independent BM25 + TF-IDF retrieval for custom PDF documents.

This module does not use the SQuAD retrieval artifacts or reader.
Contractions are preserved as single tokens to avoid false matches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from ..ingestion.pdf import PDFChunk


TOKEN_PATTERN = r"(?u)\b\w+(?:['’]\w+)?\b"


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens while preserving contractions."""
    return re.findall(TOKEN_PATTERN, text.lower())


def _normalize_max(scores: np.ndarray) -> np.ndarray:
    """Scale non-negative scores to the range 0..1."""
    scores = np.maximum(
        np.asarray(scores, dtype=np.float64),
        0.0,
    )

    maximum = float(scores.max()) if scores.size else 0.0

    if maximum <= 1e-12:
        return np.zeros_like(scores)

    return scores / maximum


@dataclass(frozen=True)
class PDFSearchResult:
    """A retrieved PDF passage and its retrieval scores."""

    chunk: PDFChunk
    score: float
    bm25_score: float
    tfidf_score: float


class PDFRetriever:
    """Search page-aware PDF chunks using BM25 and TF-IDF."""

    def __init__(self, bm25_weight: float = 0.65):
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError(
                "bm25_weight must be between 0 and 1."
            )

        self.bm25_weight = bm25_weight
        self.chunks: list[PDFChunk] = []
        self._fitted = False

    def fit(self, chunks: Iterable[PDFChunk]) -> "PDFRetriever":
        """Build the searchable index from PDF chunks."""
        items = list(chunks)

        if not items:
            raise ValueError(
                "Cannot index an empty PDF collection."
            )

        valid_items = []
        tokenized = []

        for chunk in items:
            tokens = tokenize(chunk.text)

            if tokens:
                valid_items.append(chunk)
                tokenized.append(tokens)

        if not valid_items:
            raise ValueError(
                "The PDF collection contains no searchable words."
            )

        self.chunks = valid_items

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=TOKEN_PATTERN,
            sublinear_tf=True,
            norm="l2",
        )

        try:
            self.matrix = self.vectorizer.fit_transform(
                [chunk.text for chunk in self.chunks]
            )

        except ValueError as exc:
            raise ValueError(
                f"Could not build the PDF text index: {exc}"
            ) from exc

        self.bm25 = BM25Okapi(tokenized)
        self._fitted = True

        return self

    def search(
        self,
        question: str,
        k: int = 5,
    ) -> list[PDFSearchResult]:
        """Return the best matching PDF passages.

        The combined score is a relative retrieval score, not a
        calibrated probability that an answer is correct.
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit() before search()."
            )

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if k < 1:
            raise ValueError(
                "k must be at least 1."
            )

        query_tokens = tokenize(question)

        if not query_tokens:
            return []

        bm25_raw = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=np.float64,
        )

        bm25_scores = _normalize_max(bm25_raw)

        query_vector = self.vectorizer.transform(
            [question]
        )

        tfidf_raw = np.asarray(
            (self.matrix @ query_vector.T).toarray(),
            dtype=np.float64,
        ).ravel()

        tfidf_scores = _normalize_max(tfidf_raw)

        if (
            float(bm25_scores.max()) <= 1e-12
            and float(tfidf_scores.max()) <= 1e-12
        ):
            return []

        weight = self.bm25_weight

        combined = (
            weight * bm25_scores
            + (1.0 - weight) * tfidf_scores
        )

        order = np.argsort(
            -combined,
            kind="stable",
        )[:k]

        return [
            PDFSearchResult(
                chunk=self.chunks[int(index)],
                score=float(combined[index]),
                bm25_score=float(bm25_scores[index]),
                tfidf_score=float(tfidf_scores[index]),
            )
            for index in order
            if combined[index] > 0.0
        ]
