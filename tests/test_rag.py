"""Tests for the CMS NCCI RAG system."""

import pytest
from pathlib import Path

from src.pdf_extractor import extract_pdf
from src.chunker import chunk_pages, Chunk, extract_codes, extract_code_ranges
from src.retriever import VectorlessRetriever, CodeRangeValidator
from src.rag_engine import CMSRAGEngine


PDF_PATH = Path(__file__).parent.parent / "data" / "2026-ncci-medicare-policy-manual-all-chapters.pdf"
CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"


class TestCodeExtraction:
    def test_extract_cpt_codes(self):
        text = "CPT codes 10021, 10022, and 99213 are commonly used."
        codes = extract_codes(text)
        assert "10021" in codes
        assert "10022" in codes
        assert "99213" in codes

    def test_extract_hcpcs_codes(self):
        text = "HCPCS codes G0008 and J0120 are applicable."
        codes = extract_codes(text)
        assert "G0008" in codes
        assert "J0120" in codes

    def test_extract_code_ranges(self):
        text = "Codes 10000-10999 and 99201-99215 apply."
        ranges = extract_code_ranges(text)
        assert ("10000", "10999") in ranges
        assert ("99201", "99215") in ranges


class TestChunk:
    def test_contains_code(self):
        chunk = Chunk(chunk_id=0, text="test", codes=["10021", "10022"])
        assert chunk.contains_code("10021")
        assert not chunk.contains_code("99999")

    def test_contains_code_in_range(self):
        chunk = Chunk(
            chunk_id=0,
            text="test",
            codes=["10021", "10022"],
            code_ranges=[("10000", "10999")],
        )
        assert chunk.contains_code_in_range("10000", "10030")
        assert chunk.contains_code_in_range("10020", "10025")
        assert not chunk.contains_code_in_range("99000", "99999")

    def test_range_overlap(self):
        assert Chunk._ranges_overlap("10000", "10999", "10500", "11000")
        assert not Chunk._ranges_overlap("10000", "10999", "20000", "20999")


class TestRetriever:
    @pytest.fixture
    def sample_chunks(self):
        return [
            Chunk(
                chunk_id=0,
                text="Anesthesia services for surgical procedures codes 00100-01999",
                codes=["00100"],
                code_ranges=[("00100", "01999")],
                chapter="Chapter II: Anesthesia",
            ),
            Chunk(
                chunk_id=1,
                text="Evaluation and management office visit codes 99201-99215",
                codes=["99201", "99215"],
                code_ranges=[("99201", "99215")],
                chapter="Chapter I: General",
            ),
            Chunk(
                chunk_id=2,
                text="Bundling rules prevent separate reporting of procedures",
                codes=[],
                code_ranges=[],
                chapter="Chapter I: General",
            ),
        ]

    def test_bm25_retrieval(self, sample_chunks):
        retriever = VectorlessRetriever(sample_chunks)
        results = retriever.retrieve("bundling rules", top_k=3)
        assert len(results) > 0
        assert results[0].chunk.chunk_id == 2

    def test_code_range_boost(self, sample_chunks):
        retriever = VectorlessRetriever(sample_chunks)
        results = retriever.retrieve("codes 00100-00200", top_k=3)
        assert len(results) > 0
        # Chunk with matching code range should score higher
        top_result = results[0]
        assert "code_range" in top_result.match_reason or "text_range" in top_result.match_reason


class TestCodeRangeValidator:
    @pytest.fixture
    def validator(self):
        chunks = [
            Chunk(chunk_id=0, text="", codes=["10021", "10022", "99213"]),
            Chunk(chunk_id=1, text="", codes=["50000"], code_ranges=[("50000", "59999")]),
        ]
        return CodeRangeValidator(chunks)

    def test_valid_range(self, validator):
        result = validator.validate_range("10020", "10025")
        assert result["valid"] is True
        assert "10021" in result["codes_found"]
        assert "10022" in result["codes_found"]

    def test_invalid_range(self, validator):
        result = validator.validate_range("80000", "80010")
        assert result["valid"] is False
        assert len(result["codes_found"]) == 0


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="Corpus not built yet")
class TestRAGEngine:
    @pytest.fixture
    def engine(self):
        e = CMSRAGEngine()
        e.load_corpus(CORPUS_PATH)
        return e

    def test_load_corpus(self, engine):
        assert engine.is_loaded
        assert len(engine.chunks) > 0

    def test_query_basic(self, engine):
        response = engine.query("bundling rules for procedures")
        assert len(response.answer_context) > 0

    def test_query_code_range(self, engine):
        response = engine.query("codes 10000-10999")
        assert response.code_validation is not None
        assert response.code_validation["valid"] is True

    def test_query_nonexistent_range(self, engine):
        response = engine.query("codes 80000-80010")
        assert response.code_validation is not None
