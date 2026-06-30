"""Core RAG engine that orchestrates extraction, chunking, and retrieval."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from .pdf_extractor import extract_pdf
from .chunker import chunk_pages, Chunk
from .retriever import VectorlessRetriever, CodeRangeValidator, RetrievalResult


@dataclass
class RAGResponse:
    """Response from the RAG system."""

    query: str
    answer_context: list[dict]
    code_validation: dict | None = None
    metadata: dict | None = None


class CMSRAGEngine:
    """
    Vector-less RAG engine for the 2026 NCCI Medicare Policy Manual.

    Architecture:
    - PDF text extraction via PyMuPDF
    - Chunking with configurable overlap
    - Metadata enrichment (chapter, section, CPT/HCPCS codes)
    - BM25 retrieval with code-aware boosting
    - No vector database required
    """

    def __init__(
        self,
        pdf_path: str | Path | None = None,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.retriever: VectorlessRetriever | None = None
        self.code_validator: CodeRangeValidator | None = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, pdf_path: str | Path | None = None) -> dict:
        """
        Load and index the PDF document.

        Returns metadata about the indexing process.
        """
        if pdf_path:
            self.pdf_path = Path(pdf_path)

        if not self.pdf_path or not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        start_time = time.time()

        # Step 1: Extract text from PDF
        pages = extract_pdf(self.pdf_path)

        # Step 2: Chunk with overlap and enrich metadata
        self.chunks = chunk_pages(
            pages, chunk_size=self.chunk_size, overlap=self.overlap
        )

        # Step 3: Build BM25 index
        self.retriever = VectorlessRetriever(self.chunks)
        self.code_validator = CodeRangeValidator(self.chunks)

        self._is_loaded = True
        elapsed = time.time() - start_time

        return {
            "status": "loaded",
            "pages_extracted": len(pages),
            "chunks_created": len(self.chunks),
            "total_codes_found": len(self.code_validator.all_codes),
            "total_ranges_found": len(self.code_validator.all_ranges),
            "load_time_seconds": round(elapsed, 2),
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        include_code_validation: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            query: Natural language question, optionally referencing code ranges.
            top_k: Number of context chunks to retrieve.
            include_code_validation: Whether to validate code ranges in the query.

        Returns:
            RAGResponse with retrieved context and metadata.
        """
        if not self._is_loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, top_k=top_k)

        # Format context
        answer_context = []
        for r in results:
            answer_context.append(
                {
                    "text": r.chunk.text,
                    "score": round(r.score, 4),
                    "match_reason": r.match_reason,
                    "chapter": r.chunk.chapter,
                    "section": r.chunk.section,
                    "page_numbers": r.chunk.page_numbers,
                    "codes_in_chunk": r.chunk.codes[:10],
                    "code_ranges_in_chunk": [
                        f"{s}-{e}" for s, e in r.chunk.code_ranges
                    ],
                }
            )

        # Code range validation
        code_validation = None
        if include_code_validation:
            range_start, range_end = self.retriever._parse_code_range_from_query(query)
            if range_start and range_end:
                code_validation = self.code_validator.validate_range(
                    range_start, range_end
                )

        return RAGResponse(
            query=query,
            answer_context=answer_context,
            code_validation=code_validation,
            metadata={
                "total_chunks_searched": len(self.chunks),
                "results_returned": len(results),
            },
        )

    def get_chunk_stats(self) -> dict:
        """Get statistics about the indexed corpus."""
        if not self._is_loaded:
            return {"status": "not_loaded"}

        chapters = set(c.chapter for c in self.chunks if c.chapter)
        return {
            "total_chunks": len(self.chunks),
            "total_codes": len(self.code_validator.all_codes) if self.code_validator else 0,
            "total_ranges": len(self.code_validator.all_ranges) if self.code_validator else 0,
            "chapters": sorted(chapters),
            "sample_codes": sorted(self.code_validator.all_codes)[:20] if self.code_validator else [],
        }

    def save_corpus(self, output_path: str | Path) -> None:
        """Save the chunk corpus to JSON for inspection or caching."""
        output_path = Path(output_path)
        corpus_data = []
        for chunk in self.chunks:
            corpus_data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_numbers": chunk.page_numbers,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "codes": chunk.codes,
                    "code_ranges": [f"{s}-{e}" for s, e in chunk.code_ranges],
                }
            )
        output_path.write_text(json.dumps(corpus_data, indent=2))

    def load_corpus(self, corpus_path: str | Path) -> dict:
        """Load a pre-built corpus from JSON (skip PDF extraction)."""
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        start_time = time.time()
        data = json.loads(corpus_path.read_text())

        self.chunks = []
        for item in data:
            code_ranges = []
            for cr in item.get("code_ranges", []):
                parts = cr.split("-")
                if len(parts) == 2:
                    code_ranges.append((parts[0], parts[1]))

            self.chunks.append(
                Chunk(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    page_numbers=item.get("page_numbers", []),
                    chapter=item.get("chapter", ""),
                    section=item.get("section", ""),
                    codes=item.get("codes", []),
                    code_ranges=code_ranges,
                )
            )

        self.retriever = VectorlessRetriever(self.chunks)
        self.code_validator = CodeRangeValidator(self.chunks)
        self._is_loaded = True

        elapsed = time.time() - start_time
        return {
            "status": "loaded_from_corpus",
            "chunks_loaded": len(self.chunks),
            "total_codes": len(self.code_validator.all_codes),
            "load_time_seconds": round(elapsed, 2),
        }
