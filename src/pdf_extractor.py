"""PDF text extraction with page-level metadata."""

import re
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""

    page_number: int
    text: str
    chapter: str = ""
    section: str = ""


def extract_pdf(pdf_path: str | Path) -> list[PageContent]:
    """Extract text from all pages of the PDF with basic metadata."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageContent] = []
    doc = fitz.open(str(pdf_path))

    current_chapter = ""
    current_section = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Detect chapter headings
        chapter_match = re.search(
            r"Chapter\s+(\d+|[IVXLC]+)[:\s\-]+(.+?)(?:\n|$)", text, re.IGNORECASE
        )
        if chapter_match:
            current_chapter = f"Chapter {chapter_match.group(1)}: {chapter_match.group(2).strip()}"

        # Detect section headings
        section_match = re.search(
            r"(?:Section|§)\s*([\d.]+)[:\s\-]+(.+?)(?:\n|$)", text, re.IGNORECASE
        )
        if section_match:
            current_section = f"Section {section_match.group(1)}: {section_match.group(2).strip()}"

        pages.append(
            PageContent(
                page_number=page_num + 1,
                text=text,
                chapter=current_chapter,
                section=current_section,
            )
        )

    doc.close()
    return pages
