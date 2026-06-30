"""Text chunking with overlap and metadata enrichment."""

import re
from dataclasses import dataclass, field

from .pdf_extractor import PageContent


@dataclass
class Chunk:
    """A text chunk with enriched metadata."""

    chunk_id: int
    text: str
    page_numbers: list[int] = field(default_factory=list)
    chapter: str = ""
    section: str = ""
    codes: list[str] = field(default_factory=list)
    code_ranges: list[tuple[str, str]] = field(default_factory=list)

    def contains_code(self, code: str) -> bool:
        """Check if this chunk references a specific code."""
        if code in self.codes:
            return True
        for range_start, range_end in self.code_ranges:
            if self._code_in_range(code, range_start, range_end):
                return True
        return False

    def contains_code_in_range(self, start_code: str, end_code: str) -> bool:
        """Check if this chunk has any overlap with a code range."""
        for code in self.codes:
            if self._code_in_range(code, start_code, end_code):
                return True
        for range_start, range_end in self.code_ranges:
            if self._ranges_overlap(start_code, end_code, range_start, range_end):
                return True
        return False

    @staticmethod
    def _extract_numeric(code: str) -> int | None:
        """Extract numeric value from a code string for comparison."""
        digits = re.sub(r"[^\d]", "", code)
        return int(digits) if digits else None

    @classmethod
    def _code_in_range(cls, code: str, range_start: str, range_end: str) -> bool:
        """Check if a code falls within a range."""
        code_num = cls._extract_numeric(code)
        start_num = cls._extract_numeric(range_start)
        end_num = cls._extract_numeric(range_end)
        if code_num is None or start_num is None or end_num is None:
            return False
        return start_num <= code_num <= end_num

    @classmethod
    def _ranges_overlap(
        cls, start1: str, end1: str, start2: str, end2: str
    ) -> bool:
        """Check if two code ranges overlap."""
        s1 = cls._extract_numeric(start1)
        e1 = cls._extract_numeric(end1)
        s2 = cls._extract_numeric(start2)
        e2 = cls._extract_numeric(end2)
        if any(v is None for v in [s1, e1, s2, e2]):
            return False
        return s1 <= e2 and s2 <= e1


# Regex patterns for CPT/HCPCS codes
CODE_PATTERN = re.compile(
    r"\b(\d{5})\b"  # 5-digit CPT codes
    r"|"
    r"\b([A-Z]\d{4})\b"  # HCPCS Level II codes (letter + 4 digits)
)

# Pattern for code ranges like "10000-10999" or "99201-99215"
CODE_RANGE_PATTERN = re.compile(
    r"\b(\d{5})\s*[-–—]\s*(\d{5})\b"
    r"|"
    r"\b([A-Z]\d{4})\s*[-–—]\s*([A-Z]\d{4})\b"
)


def extract_codes(text: str) -> list[str]:
    """Extract individual CPT/HCPCS codes from text."""
    codes = set()
    for match in CODE_PATTERN.finditer(text):
        code = match.group(1) or match.group(2)
        if code:
            codes.add(code)
    return sorted(codes)


def extract_code_ranges(text: str) -> list[tuple[str, str]]:
    """Extract code ranges from text."""
    ranges = []
    for match in CODE_RANGE_PATTERN.finditer(text):
        start = match.group(1) or match.group(3)
        end = match.group(2) or match.group(4)
        if start and end:
            ranges.append((start, end))
    return ranges


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk extracted pages into overlapping text segments with metadata.

    Args:
        pages: List of extracted page contents.
        chunk_size: Target characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of enriched Chunk objects.
    """
    chunks: list[Chunk] = []
    chunk_id = 0

    # Concatenate all text with page boundary markers
    segments: list[tuple[str, int, str, str]] = []
    for page in pages:
        segments.append((page.text, page.page_number, page.chapter, page.section))

    # Build chunks with overlap
    buffer = ""
    buffer_pages: list[int] = []
    buffer_chapter = ""
    buffer_section = ""

    for text, page_num, chapter, section in segments:
        buffer += text
        buffer_pages.append(page_num)
        if chapter:
            buffer_chapter = chapter
        if section:
            buffer_section = section

        while len(buffer) >= chunk_size:
            chunk_text = buffer[:chunk_size]

            # Try to break at a sentence boundary
            last_period = chunk_text.rfind(".")
            last_newline = chunk_text.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                chunk_text = buffer[: break_point + 1]

            codes = extract_codes(chunk_text)
            code_ranges = extract_code_ranges(chunk_text)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    page_numbers=sorted(set(buffer_pages)),
                    chapter=buffer_chapter,
                    section=buffer_section,
                    codes=codes,
                    code_ranges=code_ranges,
                )
            )
            chunk_id += 1

            # Advance buffer with overlap
            advance = len(chunk_text) - overlap
            buffer = buffer[advance:]
            # Keep only relevant pages
            buffer_pages = [page_num]

    # Handle remaining text
    if buffer.strip():
        codes = extract_codes(buffer)
        code_ranges = extract_code_ranges(buffer)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=buffer.strip(),
                page_numbers=sorted(set(buffer_pages)),
                chapter=buffer_chapter,
                section=buffer_section,
                codes=codes,
                code_ranges=code_ranges,
            )
        )

    return chunks
