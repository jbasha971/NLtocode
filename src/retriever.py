"""Vector-less retrieval using BM25 and metadata-based filtering."""

import re
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from .chunker import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance score."""

    chunk: Chunk
    score: float
    match_reason: str = ""


class VectorlessRetriever:
    """
    BM25-based retriever with code-aware filtering.

    Retrieval strategy:
    1. If query contains code ranges, filter chunks by code overlap first.
    2. Apply BM25 scoring on the filtered (or full) corpus.
    3. Combine code-match bonus with BM25 score for final ranking.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Tokenize chunks for BM25
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _parse_code_range_from_query(
        self, query: str
    ) -> tuple[str | None, str | None]:
        """Extract code range from a query like 'codes 1-10' or '10021-10035'."""
        # Pattern: "codes X-Y" or "code range X-Y" or just "X-Y" with numbers
        range_match = re.search(
            r"(?:codes?|range|cpt)?\s*(\d+)\s*[-–—to]+\s*(\d+)",
            query,
            re.IGNORECASE,
        )
        if range_match:
            start = range_match.group(1)
            end = range_match.group(2)
            # Pad to 5 digits if they look like CPT codes
            if len(start) <= 5:
                start = start.zfill(5)
            if len(end) <= 5:
                end = end.zfill(5)
            return start, end
        return None, None

    def _parse_single_code_from_query(self, query: str) -> str | None:
        """Extract a single code from the query."""
        code_match = re.search(r"\b(\d{5})\b|\b([A-Z]\d{4})\b", query)
        if code_match:
            return code_match.group(1) or code_match.group(2)
        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        code_boost: float = 2.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query, optionally with code references.
            top_k: Maximum results to return.
            code_boost: Score multiplier for chunks matching code criteria.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        results: list[RetrievalResult] = []

        # Parse code references from query
        range_start, range_end = self._parse_code_range_from_query(query)
        single_code = self._parse_single_code_from_query(query)

        # BM25 scoring
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        for idx, chunk in enumerate(self.chunks):
            score = float(bm25_scores[idx])
            match_reason = "bm25"

            # Code range matching
            if range_start and range_end:
                if chunk.contains_code_in_range(range_start, range_end):
                    score *= code_boost
                    match_reason = f"code_range({range_start}-{range_end})+bm25"
                elif self._chunk_text_mentions_range(chunk, range_start, range_end):
                    score *= (code_boost * 0.7)
                    match_reason = f"text_range_mention({range_start}-{range_end})+bm25"

            # Single code matching
            elif single_code:
                if chunk.contains_code(single_code):
                    score *= code_boost
                    match_reason = f"code({single_code})+bm25"

            if score > 0:
                results.append(
                    RetrievalResult(chunk=chunk, score=score, match_reason=match_reason)
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # If code range was specified but no direct matches found,
        # return best BM25 matches with a note
        if range_start and range_end:
            code_matches = [
                r for r in results if "code_range" in r.match_reason or "text_range" in r.match_reason
            ]
            if not code_matches:
                # Add context about the range not being directly found
                for r in results[:top_k]:
                    r.match_reason = f"no_direct_code_match({range_start}-{range_end}),nearest_bm25"

        return results[:top_k]

    @staticmethod
    def _chunk_text_mentions_range(
        chunk: Chunk, start: str, end: str
    ) -> bool:
        """Check if chunk text mentions codes that might relate to the range."""
        start_num = int(re.sub(r"[^\d]", "", start) or "0")
        end_num = int(re.sub(r"[^\d]", "", end) or "0")

        for code in chunk.codes:
            code_num = int(re.sub(r"[^\d]", "", code) or "0")
            # Check if any code in chunk is within a reasonable proximity
            if start_num - 1000 <= code_num <= end_num + 1000:
                return True
        return False


class CodeRangeValidator:
    """Validates and provides context for code range queries."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Build a set of all known codes
        self.all_codes: set[str] = set()
        self.all_ranges: list[tuple[str, str]] = []
        for chunk in chunks:
            self.all_codes.update(chunk.codes)
            self.all_ranges.extend(chunk.code_ranges)

    def validate_range(self, start: str, end: str) -> dict:
        """
        Validate a code range and provide information about coverage.

        Returns a dict with:
        - valid: whether the range has any known codes
        - codes_found: specific codes found in range
        - nearest_codes: if no exact match, nearest known codes
        - coverage_info: textual description of what the range covers
        """
        start_num = int(re.sub(r"[^\d]", "", start) or "0")
        end_num = int(re.sub(r"[^\d]", "", end) or "0")

        codes_in_range = []
        for code in sorted(self.all_codes):
            code_num = int(re.sub(r"[^\d]", "", code) or "0")
            if start_num <= code_num <= end_num:
                codes_in_range.append(code)

        # Find overlapping defined ranges
        overlapping_ranges = []
        for r_start, r_end in self.all_ranges:
            rs = int(re.sub(r"[^\d]", "", r_start) or "0")
            re_ = int(re.sub(r"[^\d]", "", r_end) or "0")
            if rs <= end_num and start_num <= re_:
                overlapping_ranges.append((r_start, r_end))

        if codes_in_range or overlapping_ranges:
            return {
                "valid": True,
                "codes_found": codes_in_range[:20],  # Limit output
                "overlapping_ranges": overlapping_ranges[:10],
                "coverage_info": (
                    f"Found {len(codes_in_range)} specific codes and "
                    f"{len(overlapping_ranges)} overlapping ranges in {start}-{end}."
                ),
            }

        # No direct match - find nearest codes
        nearest_below = []
        nearest_above = []
        for code in sorted(self.all_codes):
            code_num = int(re.sub(r"[^\d]", "", code) or "0")
            if code_num < start_num:
                nearest_below.append(code)
            elif code_num > end_num:
                nearest_above.append(code)

        return {
            "valid": False,
            "codes_found": [],
            "overlapping_ranges": [],
            "nearest_below": nearest_below[-5:] if nearest_below else [],
            "nearest_above": nearest_above[:5] if nearest_above else [],
            "coverage_info": (
                f"No codes found directly in range {start}-{end}. "
                f"Nearest codes below: {nearest_below[-3:] if nearest_below else 'none'}. "
                f"Nearest codes above: {nearest_above[:3] if nearest_above else 'none'}."
            ),
        }
