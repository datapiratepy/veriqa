
import pytest

import veriqa.ingestion.pdf as pdf_ingestion
import veriqa.pdf_service as pdf_service

from veriqa.ingestion.pdf import PDFChunk, extract_pdf_chunks
from veriqa.pdf_service import PDFQAService
from veriqa.retrieval.pdf import tokenize


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------

class FakePage:
    """Simulated PDF page for predictable extraction tests."""

    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    """Simulated two-page PDF."""

    def __init__(self, stream, strict=False):
        self.is_encrypted = False
        self.pages = [
            FakePage(
                "Supervised learning uses labeled examples "
                "to train a predictive model."
            ),
            FakePage(
                "Unsupervised learning discovers structure "
                "in unlabeled data."
            ),
        ]


def sample_chunks():
    """Return sample page-aware PDF chunks."""

    return [
        PDFChunk(
            chunk_id=0,
            document="lecture.pdf",
            page=1,
            text=(
                "Supervised learning uses labeled examples "
                "to train a predictive model."
            ),
        ),
        PDFChunk(
            chunk_id=1,
            document="lecture.pdf",
            page=2,
            text=(
                "Unsupervised learning discovers structure "
                "in unlabeled data."
            ),
        ),
    ]


# ---------------------------------------------------------------------
# PDF tokenizer tests
# ---------------------------------------------------------------------

def test_tokenizer_preserves_contractions():
    """Won't must not be tokenized as won and t."""

    tokens = tokenize(
        "Things won't transfer if they appear elsewhere."
    )

    assert "won't" in tokens
    assert "won" not in tokens
    assert "t" not in tokens


def test_keyword_match_requirement():
    """Question terms should require sufficient overlap."""

    from veriqa.pdf_service import _minimum_keyword_matches

    assert _minimum_keyword_matches(1) == 1
    assert _minimum_keyword_matches(2) == 2
    assert _minimum_keyword_matches(3) == 2
    assert _minimum_keyword_matches(4) == 2
    assert _minimum_keyword_matches(5) == 3


# ---------------------------------------------------------------------
# PDF ingestion tests
# ---------------------------------------------------------------------

def test_pdf_extraction_preserves_page_numbers(monkeypatch):
    """Extracted chunks should retain original PDF page numbers."""

    monkeypatch.setattr(
        pdf_ingestion,
        "PdfReader",
        FakeReader,
    )

    chunks = extract_pdf_chunks([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    assert len(chunks) == 2

    assert chunks[0].document == "lecture.pdf"
    assert chunks[0].page == 1
    assert "Supervised learning" in chunks[0].text

    assert chunks[1].document == "lecture.pdf"
    assert chunks[1].page == 2
    assert "Unsupervised learning" in chunks[1].text


def test_pdf_extraction_rejects_empty_documents(monkeypatch):
    """PDFs without extractable text should raise a clear error."""

    class EmptyReader:
        def __init__(self, stream, strict=False):
            self.is_encrypted = False
            self.pages = [FakePage("")]

    monkeypatch.setattr(
        pdf_ingestion,
        "PdfReader",
        EmptyReader,
    )

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_chunks([
            ("empty.pdf", b"fake-pdf-content")
        ])


def test_pdf_extraction_removes_header_and_captures_printed_page(
    monkeypatch,
):
    """Recognizable headers should be removed and printed pages captured."""

    class HeaderReader:
        def __init__(self, stream, strict=False):
            self.is_encrypted = False
            self.pages = [
                FakePage(
                    "DEPARTMENT OF CSE MACHINE LEARNING Page 2\n"
                    "Supervised Learning is guided by a teacher."
                )
            ]

    monkeypatch.setattr(
        pdf_ingestion,
        "PdfReader",
        HeaderReader,
    )

    chunks = extract_pdf_chunks([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    assert len(chunks) == 1

    assert chunks[0].document == "lecture.pdf"
    assert chunks[0].page == 1
    assert chunks[0].printed_page == 2

    assert "DEPARTMENT OF CSE" not in chunks[0].text
    assert "Page 2" not in chunks[0].text
    assert "Supervised Learning" in chunks[0].text


# ---------------------------------------------------------------------
# PDF question-answering service tests
# ---------------------------------------------------------------------

def test_pdf_service_returns_answer_and_citation(monkeypatch):
    """The service should return an answer with its source citation."""

    monkeypatch.setattr(
        pdf_service,
        "extract_pdf_chunks",
        lambda *args, **kwargs: sample_chunks(),
    )

    service = PDFQAService()

    stats = service.index_documents([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    assert service.is_ready

    assert stats["documents"] == ["lecture.pdf"]
    assert stats["pages"] == 2
    assert stats["chunks"] == 2

    result = service.ask(
        "What is supervised learning?",
        k=2,
    )

    assert result["abstained"] is False
    assert "labeled examples" in result["answer"]

    assert result["source"]["document"] == "lecture.pdf"
    assert result["source"]["page"] == 1

    assert result["match_score"] is not None
    assert len(result["evidence"]) > 0

    assert result["answer_method"] == "extractive sentence selection"
    assert result["retrieval_method"] == "BM25 + TF-IDF"


def test_pdf_service_abstains_without_matching_evidence(monkeypatch):
    """The service should abstain when no relevant evidence exists."""

    monkeypatch.setattr(
        pdf_service,
        "extract_pdf_chunks",
        lambda *args, **kwargs: sample_chunks(),
    )

    service = PDFQAService()

    service.index_documents([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    result = service.ask(
        "Who won the lunar championship?",
        k=2,
    )

    assert result["abstained"] is True
    assert result["answer"] is None
    assert result["source"] is None
    assert result["reason"]
    assert result["match_score"] is None


def test_pdf_service_abstains_on_weak_keyword_overlap(monkeypatch):
    """One coincidental keyword must not produce an answer."""

    weak_chunks = [
        PDFChunk(
            chunk_id=0,
            document="lecture.pdf",
            page=65,
            text=(
                "Things we learn about the sequence won't transfer "
                "if they appear at different points in the sequence. "
                "The championship is discussed in this section."
            ),
            printed_page=58,
        )
    ]

    monkeypatch.setattr(
        pdf_service,
        "extract_pdf_chunks",
        lambda *args, **kwargs: weak_chunks,
    )

    service = PDFQAService()

    service.index_documents([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    result = service.ask(
        "Who won the lunar championship?",
        k=1,
    )

    assert result["abstained"] is True
    assert result["answer"] is None
    assert result["source"] is None
    assert result["match_score"] is None

    assert result["reason"]
    assert len(result["evidence"]) > 0



def test_pdf_service_abstains_when_only_contraction_matches(monkeypatch):
    """Won't must not create a false answer for the word won.

    Retrieved evidence may still be displayed for inspection,
    even when the answer service correctly abstains.
    """

    contraction_chunks = [
        PDFChunk(
            chunk_id=0,
            document="lecture.pdf",
            page=65,
            text=(
                "Things we learn about the sequence won't transfer "
                "if they appear at different points in the sequence."
            ),
            printed_page=58,
        )
    ]

    monkeypatch.setattr(
        pdf_service,
        "extract_pdf_chunks",
        lambda *args, **kwargs: contraction_chunks,
    )

    service = PDFQAService()

    service.index_documents([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    result = service.ask(
        "Who won the lunar championship?",
        k=1,
    )

    # The system must abstain.
    assert result["abstained"] is True
    assert result["answer"] is None
    assert result["source"] is None
    assert result["match_score"] is None

    # Retrieved passages are retained for transparency.
    assert len(result["evidence"]) > 0

    assert result["reason"]


def test_pdf_service_answers_from_following_question_heading(monkeypatch):
    """A matching question heading should return its explanation."""

    heading_chunks = [
        PDFChunk(
            chunk_id=0,
            document="lecture.pdf",
            page=9,
            text=(
                "What is Unsupervised Learning? "
                "The model learns through observation and "
                "finds structures in the data."
            ),
            printed_page=2,
        )
    ]

    monkeypatch.setattr(
        pdf_service,
        "extract_pdf_chunks",
        lambda *args, **kwargs: heading_chunks,
    )

    service = PDFQAService()

    service.index_documents([
        ("lecture.pdf", b"fake-pdf-content")
    ])

    result = service.ask(
        "What is Unsupervised Learning?",
        k=1,
    )

    assert result["abstained"] is False

    assert result["answer"] == (
        "The model learns through observation and "
        "finds structures in the data."
    )

    assert result["answer"] != (
        "What is Unsupervised Learning?"
    )

    assert result["source"]["page"] == 9
    assert result["source"]["printed_page"] == 2
    assert result["answer_source"] == "following question heading"
