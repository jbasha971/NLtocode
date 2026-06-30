# CMS NCCI Medicare Policy RAG (Vector-less)

A vector-less Retrieval-Augmented Generation system for the **2026 NCCI Medicare Policy Manual**. Uses BM25 keyword retrieval with code-aware boosting — no vector database required.

## Architecture

```
PDF text extraction → Chunking with overlap → Metadata enrichment → BM25 Index
                                                (chapter, section, codes)
```

**Key features:**
- Answers NCCI policy questions using the official source document
- Reduces hallucination risk by grounding answers in retrieved evidence
- No operational overhead of a vector database
- Fallback-friendly retrieval architecture for local demos and production-style evaluation
- Code range validation: queries like "codes 1-10" intelligently find relevant content even if exact codes aren't present

## Document Source

- **Document**: `2026-ncci-medicare-policy-manual-all-chapters.pdf`
- **Source**: https://www.cms.gov/files/document/2026-ncci-medicare-policy-manual-all-chapters.pdf

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download the PDF (place in data/)
mkdir -p data
wget "https://www.cms.gov/files/document/2026-ncci-medicare-policy-manual-all-chapters.pdf" -O data/2026-ncci-medicare-policy-manual-all-chapters.pdf

# Start the API server
python run.py
```

The server auto-loads the PDF on startup and builds the BM25 index (~0.5s).

## API Endpoints

### `POST /query`
Query the NCCI manual with natural language or code references.

```json
{
  "query": "bundling rules for surgical procedures codes 10000-10999",
  "top_k": 5,
  "include_code_validation": true
}
```

### `POST /validate-code-range`
Check if a code range exists and find nearest codes.

```json
{
  "start_code": "10000",
  "end_code": "10999"
}
```

### `GET /health`
Health check with engine status.

### `GET /stats`
Corpus statistics (total chunks, codes, chapters).

### `GET /chunks/{chunk_id}`
Retrieve a specific chunk by ID.

### `POST /load`
Reload the document with custom chunk settings.

## Agent Integration

The `/query` endpoint is designed for agent integration:

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the bundling rules for codes 99201-99215?",
    "top_k": 5
})

result = response.json()
# result["answer_context"] - list of relevant chunks with metadata
# result["code_validation"] - validation info about the code range
# result["metadata"] - retrieval stats
```

### Code Range Handling

When querying a range (e.g., "codes 1-10"):
1. The system validates whether codes exist in that range
2. Returns matching chunks with code-boosted relevance
3. If no direct match: returns nearest relevant content + validation showing closest available codes

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Technical Details

- **PDF Extraction**: PyMuPDF (fitz) for text extraction with page metadata
- **Chunking**: 1000-char chunks with 200-char overlap, sentence-boundary aware
- **Code Detection**: Regex-based CPT (5-digit) and HCPCS (letter+4 digits) extraction
- **Retrieval**: BM25Okapi with code-aware score boosting (2x for code matches)
- **Metadata**: Chapter, section, page numbers, extracted codes and ranges per chunk
