"""PDF text extraction and chunking with error handling."""
import os
import hashlib
from typing import List, Dict
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def extract_text_from_pdf(pdf_path: str | Path) -> List[Dict]:
    """
    Extract text from PDF with page-level metadata.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'text', 'page', and 'source' keys

    Raises:
        PDFProcessingError: If extraction fails after retries
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    text_segments = []

    logger.info("extracting_pdf", file=filename)

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            logger.info("pdf_pages_found", file=filename, pages=total_pages)

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    text_segments.append({
                        "text": text.strip(),
                        "page": page_num,
                        "source": f"{filename}:page_{page_num}",
                        "total_pages": total_pages
                    })
                else:
                    logger.warning("empty_page", file=filename, page=page_num)
    except Exception as e:
        logger.error("pdf_extraction_failed", file=filename, error=str(e))
        raise PDFProcessingError(f"Failed to extract text from {filename}: {str(e)}")

    if not text_segments:
        logger.warning("no_text_extracted", file=filename)
        raise PDFProcessingError(f"No extractable text found in {filename}. It may be a scanned/image PDF.")

    logger.info("extraction_complete", file=filename, segments=len(text_segments))
    return text_segments


def create_chunks(text_segments: List[Dict], filename: str) -> List[Dict]:
    """
    Split text segments into overlapping chunks.

    Args:
        text_segments: Output from extract_text_from_pdf
        filename: Name of the source PDF

    Returns:
        List of chunk dicts with 'id', 'text', and 'metadata'
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []
    for segment in text_segments:
        split_texts = splitter.split_text(segment["text"])

        for i, chunk_text in enumerate(split_texts):
            # Deterministic ID to prevent duplicates
            chunk_id = hashlib.sha256(
                f"{filename}:{segment['page']}:{i}:{chunk_text[:50]}".encode()
            ).hexdigest()[:32]

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": segment["source"],
                    "page": segment["page"],
                    "chunk_index": i,
                    "filename": filename,
                    "total_pages": segment.get("total_pages", 0)
                }
            })

    logger.info("chunking_complete", file=filename, chunks=len(chunks))
    return chunks


def process_pdf(pdf_path: str | Path) -> List[Dict]:
    """
    Full pipeline: extract text → create chunks.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of chunks ready for embedding
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    segments = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(segments, filename)
    return chunks
