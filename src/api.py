"""FastAPI application for the CMS NCCI RAG system."""

import os
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_engine import CMSRAGEngine

app = FastAPI(
    title="CMS NCCI Medicare Policy RAG",
    description=(
        "Vector-less RAG system for the 2026 NCCI Medicare Policy Manual. "
        "Uses BM25 retrieval with code-aware filtering — no vector database required."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = CMSRAGEngine()

# Default PDF path
DEFAULT_PDF_PATH = Path(__file__).parent.parent / "data" / "2026-ncci-medicare-policy-manual-all-chapters.pdf"
DEFAULT_CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str = Field(..., description="Natural language query about NCCI policy, optionally with code references")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    include_code_validation: bool = Field(default=True, description="Whether to validate code ranges")


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    query: str
    answer_context: list[dict]
    code_validation: dict | None = None
    metadata: dict | None = None


class LoadRequest(BaseModel):
    """Request body for the /load endpoint."""

    pdf_path: str | None = None
    corpus_path: str | None = None
    chunk_size: int = Field(default=1000, ge=200, le=5000)
    overlap: int = Field(default=200, ge=50, le=1000)


class CodeRangeRequest(BaseModel):
    """Request for code range validation."""

    start_code: str = Field(..., description="Start of code range (e.g., '00100' or '1')")
    end_code: str = Field(..., description="End of code range (e.g., '00999' or '10')")


@app.on_event("startup")
async def startup_load():
    """Auto-load the corpus or PDF on startup if available."""
    if DEFAULT_CORPUS_PATH.exists():
        try:
            result = engine.load_corpus(DEFAULT_CORPUS_PATH)
            print(f"[startup] Loaded corpus: {result}")
        except Exception as e:
            print(f"[startup] Failed to load corpus: {e}")
    elif DEFAULT_PDF_PATH.exists():
        try:
            engine.pdf_path = DEFAULT_PDF_PATH
            engine.chunk_size = 1000
            engine.overlap = 200
            result = engine.load()
            print(f"[startup] Loaded PDF: {result}")
            # Save corpus for faster future loads
            engine.save_corpus(DEFAULT_CORPUS_PATH)
            print(f"[startup] Saved corpus to {DEFAULT_CORPUS_PATH}")
        except Exception as e:
            print(f"[startup] Failed to load PDF: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_loaded": engine.is_loaded,
        "chunks_indexed": len(engine.chunks) if engine.is_loaded else 0,
    }


@app.post("/load", response_model=dict)
async def load_document(request: LoadRequest):
    """Load or reload the PDF document into the RAG engine."""
    try:
        engine.chunk_size = request.chunk_size
        engine.overlap = request.overlap

        if request.corpus_path:
            result = engine.load_corpus(request.corpus_path)
        else:
            pdf_path = request.pdf_path or str(DEFAULT_PDF_PATH)
            result = engine.load(pdf_path)
            # Save corpus for caching
            engine.save_corpus(DEFAULT_CORPUS_PATH)

        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the NCCI Medicare Policy Manual.

    Supports:
    - Natural language questions about NCCI policies
    - Code-specific queries (e.g., "What does code 10021 cover?")
    - Code range queries (e.g., "Explain codes 10000-10999")
    - Mixed queries (e.g., "bundling rules for codes 99201-99215")

    When a code range is queried but no exact match exists,
    the system returns the most relevant content and validates
    whether the range exists in the document.
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503, detail="Engine not loaded. Call POST /load first."
        )

    try:
        response = engine.query(
            query=request.query,
            top_k=request.top_k,
            include_code_validation=request.include_code_validation,
        )
        return QueryResponse(
            query=response.query,
            answer_context=response.answer_context,
            code_validation=response.code_validation,
            metadata=response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate-code-range")
async def validate_code_range(request: CodeRangeRequest):
    """
    Validate whether a code range exists in the NCCI manual.

    Returns information about which codes are found within the range,
    and suggests nearest available codes if the range is empty.
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503, detail="Engine not loaded. Call POST /load first."
        )

    # Normalize codes (pad to 5 digits)
    start = request.start_code.zfill(5)
    end = request.end_code.zfill(5)

    result = engine.code_validator.validate_range(start, end)
    return result


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed document corpus."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503, detail="Engine not loaded. Call POST /load first."
        )
    return engine.get_chunk_stats()


@app.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: int):
    """Get a specific chunk by ID."""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Engine not loaded.")
    if chunk_id < 0 or chunk_id >= len(engine.chunks):
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found.")

    chunk = engine.chunks[chunk_id]
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "page_numbers": chunk.page_numbers,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "codes": chunk.codes,
        "code_ranges": [f"{s}-{e}" for s, e in chunk.code_ranges],
    }
