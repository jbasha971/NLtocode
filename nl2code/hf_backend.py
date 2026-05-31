"""HuggingFace Inference API backend (free tier, no token required for many models)."""

from typing import Optional
import requests


HF_API_URL = "https://api-inference.huggingface.co/models"
DEFAULT_MODEL = "bigcode/starcoder2-3b"

FALLBACK_MODELS = [
    "bigcode/starcoder2-3b",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "bigcode/starcoderbase-1b",
]


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    token: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Optional[str]:
    """Generate code using HuggingFace free Inference API.

    Many models on HF allow free inference without a token (rate-limited).
    Pass a token for higher rate limits if available.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }

    for model_name in [model] + [m for m in FALLBACK_MODELS if m != model]:
        try:
            url = f"{HF_API_URL}/{model_name}"
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "")
                    return text.strip()
            elif resp.status_code == 503:
                # Model is loading, try next
                continue
            else:
                continue
        except Exception:
            continue

    return None
