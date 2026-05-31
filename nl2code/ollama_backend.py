"""Ollama backend for local LLM inference (free, no tokens needed)."""

import json
import requests
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"


def is_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama server is running and reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """List available models in Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Generate text using Ollama API (streaming, concatenated)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        result_parts: list[str] = []
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                result_parts.append(token)
                if chunk.get("done", False):
                    break

        return "".join(result_parts).strip()
    except Exception as e:
        print(f"[Ollama] Error: {e}")
        return None
