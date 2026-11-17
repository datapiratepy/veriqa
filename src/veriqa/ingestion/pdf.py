
"""Text extraction and page-aware chunking for custom PDF documents.

This module is separate from the SQuAD ingestion pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFChunk:
    """A searchable piece of text with its original PDF location."""

    chunk_id: int
    document: str
    page: int
    text: str
    printed_page: int | None = None


def _clean_page_text(raw_text: str) -> tuple[str, int | None]:
    """Remove recognizable uppercase page headers and capture page labels."""
    printed_page = None

    match = re.search(
        r"\bPage\s+(\d+)\b",
        raw_text[:300],
        flags=re.IGNORECASE,
    )

    if match:
        prefix = raw_text[:match.start()].strip()
        normalized_prefix = " ".join(prefix.split())

        if (
            not normalized_prefix
            or (
                normalized_prefix.isupper()
                and len(normalized_prefix) <= 120
            )
        ):
            printed_page = int(match.group(1))
            raw_text = raw_text[match.end():]

    return " ".join(raw_text.split()), printed_page


def _split_page(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split a page into overlapping character-based chunks."""
    if len(text) <= chunk_chars:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_chars, len(text))

        if end < len(text):
            boundary = text.rfind(
                " ",
                start + int(chunk_chars * 0.7),
                end,
            )

            if boundary > start:
                end = boundary

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def extract_pdf_chunks(
    pdf_files: Iterable[tuple[str, bytes]],
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[PDFChunk]:
    """Extract PDF text and create page-aware overlapping chunks."""

    if chunk_chars < 100:
        raise ValueError(
            "chunk_chars must be at least 100."
        )

    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError(
            "overlap_chars must be non-negative "
            "and smaller than chunk_chars."
        )

    chunks: list[PDFChunk] = []
    next_id = 0

    for filename, file_bytes in pdf_files:

        if not file_bytes:
            raise ValueError(
                f"{filename} is empty."
            )

        try:
            reader = PdfReader(
                BytesIO(file_bytes),
                strict=False,
            )

            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")

                if not decrypt_result:
                    raise ValueError(
                        f"{filename} is password-protected. "
                        "Password-protected PDFs are not supported."
                    )

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):

                try:
                    raw_text = page.extract_text() or ""

                except Exception as exc:
                    raise ValueError(
                        f"Could not extract text from page "
                        f"{page_number} of {filename}: {exc}"
                    ) from exc

                text, printed_page = _clean_page_text(raw_text)

                if not text:
                    continue

                page_chunks = _split_page(
                    text,
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                )

                for piece in page_chunks:
                    chunks.append(
                        PDFChunk(
                            chunk_id=next_id,
                            document=filename,
                            page=page_number,
                            text=piece,
                            printed_page=printed_page,
                        )
                    )

                    next_id += 1

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                f"Could not read {filename} as a PDF: {exc}"
            ) from exc

    if not chunks:
        raise ValueError(
            "No extractable text was found. This may be a scanned "
            "or image-only PDF. OCR is not supported in this version."
        )

    return chunks
