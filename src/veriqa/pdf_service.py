
"""Independent extractive question answering for user-uploaded PDFs.

This service does not use the SQuAD reader, trained artifacts, or
SQuAD-calibrated reliability model.

It supports:
- Page-aware PDF retrieval
- BM25 + TF-IDF search
- Extractive sentence selection
- Question-heading detection and answer extraction
- Minimum keyword-overlap requirements
- Abstention when evidence is insufficient
"""
from __future__ import annotations

import math
import re
from typing import Iterable

from .ingestion.pdf import PDFChunk, extract_pdf_chunks
from .retrieval.pdf import PDFRetriever, tokenize


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "how", "i", "in", "into", "is", "it", "its", "may",
    "might", "my", "of", "on", "or", "our", "ours", "she",
    "should", "so", "that", "the", "their", "theirs", "them",
    "there", "these", "they", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "will", "with", "would", "you",
    "your", "yours",
}

MIN_KEYWORD_COVERAGE = 0.50


def _sentences(text: str) -> list[str]:
    """Split a passage into approximate sentences."""
    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def _minimum_keyword_matches(query_term_count: int) -> int:
    """Calculate the minimum meaningful keyword overlap."""
    if query_term_count <= 0:
        return 0

    return min(
        query_term_count,
        max(
            2,
            math.ceil(
                MIN_KEYWORD_COVERAGE * query_term_count
            ),
        ),
    )


def _is_question_heading(sentence: str) -> bool:
    """Identify question-style headings in extracted PDF text."""
    return sentence.strip().endswith("?")


class PDFQAService:
    """Index PDFs and return source-grounded extractive answers."""

    def __init__(
        self,
        *,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        bm25_weight: float = 0.65,
    ):
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.bm25_weight = bm25_weight

        self.chunks: list[PDFChunk] = []
        self.retriever: PDFRetriever | None = None
        self.documents: list[str] = []

    @property
    def is_ready(self) -> bool:
        return self.retriever is not None

    def index_documents(
        self,
        pdf_files: Iterable[tuple[str, bytes]],
    ) -> dict:
        """Extract and index a new collection of PDFs."""
        chunks = extract_pdf_chunks(
            pdf_files,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
        )

        retriever = PDFRetriever(
            bm25_weight=self.bm25_weight
        ).fit(chunks)

        # Replace the existing index only after successful indexing.
        self.chunks = chunks
        self.retriever = retriever

        self.documents = sorted({
            chunk.document
            for chunk in chunks
        })

        return {
            "documents": self.documents.copy(),
            "pages": len({
                (chunk.document, chunk.page)
                for chunk in chunks
            }),
            "chunks": len(chunks),
        }

    def ask(
        self,
        question: str,
        k: int = 5,
    ) -> dict:
        """Retrieve evidence and select a sufficiently relevant answer."""
        if self.retriever is None:
            raise RuntimeError(
                "No PDFs have been indexed. Upload a PDF first."
            )

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if k < 1:
            raise ValueError(
                "k must be at least 1."
            )

        question = question.strip()

        query_terms = {
            token
            for token in tokenize(question)
            if token not in _STOPWORDS
        }

        if not query_terms:
            return self._empty_result(
                question,
                "The question does not contain searchable keywords.",
            )

        required_matches = _minimum_keyword_matches(
            len(query_terms)
        )

        results = self.retriever.search(
            question,
            k=k,
        )

        if not results:
            return self._empty_result(
                question,
                "No matching evidence was found in the uploaded PDFs.",
            )

        candidates = []

        for result in results:

            sentences = _sentences(result.chunk.text)

            for index, sentence in enumerate(sentences):

                sentence_terms = set(
                    tokenize(sentence)
                )

                overlap = query_terms.intersection(
                    sentence_terms
                )

                coverage = (
                    len(overlap) / len(query_terms)
                )

                # Question-style headings should never be returned
                # as answers. When a heading matches the query,
                # select its following explanatory sentence instead.
                if _is_question_heading(sentence):

                    if (
                        len(overlap) >= required_matches
                        and coverage >= MIN_KEYWORD_COVERAGE
                    ):

                        following_sentence = next(
                            (
                                following
                                for following in sentences[index + 1:]
                                if not _is_question_heading(following)
                            ),
                            None,
                        )

                        if following_sentence:

                            rank_score = (
                                0.75 * coverage
                                + 0.25 * result.score
                            )

                            candidates.append({
                                "sentence": following_sentence,
                                "document": result.chunk.document,
                                "page": result.chunk.page,
                                "printed_page": result.chunk.printed_page,
                                "coverage": coverage,
                                "rank_score": rank_score,
                                "overlap_terms": sorted(overlap),
                                "retrieval_score": result.score,
                                "answer_source": "following question heading",
                            })

                    # Never return a question heading as an answer.
                    continue

                # For ordinary declarative sentences, require
                # sufficient direct keyword overlap.
                if len(overlap) < required_matches:
                    continue

                if coverage < MIN_KEYWORD_COVERAGE:
                    continue

                rank_score = (
                    0.75 * coverage
                    + 0.25 * result.score
                )

                candidates.append({
                    "sentence": sentence,
                    "document": result.chunk.document,
                    "page": result.chunk.page,
                    "printed_page": result.chunk.printed_page,
                    "coverage": coverage,
                    "rank_score": rank_score,
                    "overlap_terms": sorted(overlap),
                    "retrieval_score": result.score,
                    "answer_source": "direct sentence match",
                })

        if not candidates:
            return self._empty_result(
                question,
                (
                    "Retrieved passages did not contain enough "
                    "matching keywords to support an answer. "
                    "Please rephrase the question or check "
                    "whether the information exists in the PDFs."
                ),
                results=results,
            )

        candidates.sort(
            key=lambda item: (
                item["rank_score"],
                item["coverage"],
                item["retrieval_score"],
            ),
            reverse=True,
        )

        best = candidates[0]

        evidence = [
            {
                "document": item.chunk.document,
                "page": item.chunk.page,
                "printed_page": item.chunk.printed_page,
                "score": round(item.score, 4),
                "text": item.chunk.text,
            }
            for item in results
        ]

        return {
            "question": question,
            "answer": best["sentence"],
            "abstained": False,
            "reason": "",
            "source": {
                "document": best["document"],
                "page": best["page"],
                "printed_page": best["printed_page"],
            },
            "match_score": round(
                best["rank_score"],
                4,
            ),
            "matched_terms": best["overlap_terms"],
            "answer_source": best["answer_source"],
            "evidence": evidence,
            "answer_method": "extractive sentence selection",
            "retrieval_method": "BM25 + TF-IDF",
            "warning": (
                "The answer is an extracted sentence from the PDF. "
                "The relative match score is not a calibrated "
                "probability of answer correctness."
            ),
        }

    @staticmethod
    def _empty_result(
        question: str,
        reason: str,
        results: list | None = None,
    ) -> dict:
        """Return a structured abstention result."""
        evidence = []

        if results:
            evidence = [
                {
                    "document": item.chunk.document,
                    "page": item.chunk.page,
                    "printed_page": item.chunk.printed_page,
                    "score": round(item.score, 4),
                    "text": item.chunk.text,
                }
                for item in results
            ]

        return {
            "question": question,
            "answer": None,
            "abstained": True,
            "reason": reason,
            "source": None,
            "match_score": None,
            "matched_terms": [],
            "answer_source": None,
            "evidence": evidence,
            "answer_method": "extractive sentence selection",
            "retrieval_method": "BM25 + TF-IDF",
            "warning": (
                "No answer was produced because the retrieved "
                "evidence did not meet the minimum relevance "
                "requirement. This PDF mode does not have a "
                "calibrated probability of correctness."
            ),
        }
