"""Core engine that orchestrates code generation across backends."""

from typing import Optional

from nl2code.prompts import (
    LANGUAGE_CONFIGS,
    SUPPORTED_LANGUAGES,
    build_prompt,
    build_explanation_prompt,
)
from nl2code import ollama_backend, hf_backend


class CodeGenerator:
    """Main code generation engine with automatic backend selection."""

    def __init__(
        self,
        backend: str = "auto",
        ollama_model: str = ollama_backend.DEFAULT_MODEL,
        hf_model: str = hf_backend.DEFAULT_MODEL,
        hf_token: Optional[str] = None,
    ):
        """Initialize the code generator.

        Args:
            backend: "ollama", "huggingface", or "auto" (tries ollama first).
            ollama_model: Ollama model name.
            hf_model: HuggingFace model name.
            hf_token: Optional HuggingFace token for higher rate limits.
        """
        self.backend = backend
        self.ollama_model = ollama_model
        self.hf_model = hf_model
        self.hf_token = hf_token
        self._active_backend: Optional[str] = None

    @property
    def active_backend(self) -> str:
        """Return which backend is currently being used."""
        if self._active_backend:
            return self._active_backend
        if self.backend != "auto":
            return self.backend
        return "ollama" if ollama_backend.is_ollama_available() else "huggingface"

    def get_status(self) -> dict:
        """Get status of all backends."""
        ollama_available = ollama_backend.is_ollama_available()
        ollama_models = ollama_backend.list_models() if ollama_available else []
        return {
            "ollama": {
                "available": ollama_available,
                "models": ollama_models,
            },
            "huggingface": {
                "available": True,  # always available (free tier)
            },
            "active_backend": self.active_backend,
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    def generate_code(
        self,
        task: str,
        language: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate code from a natural language task description.

        Args:
            task: Natural language description of what the code should do.
            language: Target language (python, java, cpp, csharp).
            temperature: Creativity (0.0=deterministic, 1.0=creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with keys: code, language, backend_used, success, error
        """
        if language not in SUPPORTED_LANGUAGES:
            return {
                "code": "",
                "language": language,
                "backend_used": None,
                "success": False,
                "error": f"Unsupported language: {language}. Choose from: {', '.join(SUPPORTED_LANGUAGES)}",
            }

        prompt = build_prompt(task, language)
        code = None
        backend_used = None

        # Try backends based on configuration
        if self.backend in ("auto", "ollama"):
            if ollama_backend.is_ollama_available():
                code = ollama_backend.generate(
                    prompt,
                    model=self.ollama_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if code:
                    backend_used = "ollama"

        if code is None and self.backend in ("auto", "huggingface"):
            code = hf_backend.generate(
                prompt,
                model=self.hf_model,
                token=self.hf_token,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if code:
                backend_used = "huggingface"

        if code is None:
            return {
                "code": "",
                "language": language,
                "backend_used": None,
                "success": False,
                "error": (
                    "No backend available. Please ensure Ollama is running "
                    "(run 'ollama serve' in a terminal) or check your internet "
                    "connection for HuggingFace API."
                ),
            }

        self._active_backend = backend_used
        cleaned_code = self._clean_code(code, language)

        return {
            "code": cleaned_code,
            "language": language,
            "language_name": LANGUAGE_CONFIGS[language]["name"],
            "extension": LANGUAGE_CONFIGS[language]["extension"],
            "run_command": LANGUAGE_CONFIGS[language]["run_command"],
            "backend_used": backend_used,
            "success": True,
            "error": None,
        }

    def explain_code(self, code: str, language: str) -> Optional[str]:
        """Generate an explanation for the given code."""
        prompt = build_explanation_prompt(code, language)

        if self.backend in ("auto", "ollama") and ollama_backend.is_ollama_available():
            result = ollama_backend.generate(prompt, model=self.ollama_model)
            if result:
                return result

        if self.backend in ("auto", "huggingface"):
            result = hf_backend.generate(
                prompt, model=self.hf_model, token=self.hf_token
            )
            if result:
                return result

        return None

    @staticmethod
    def _clean_code(code: str, language: str) -> str:
        """Remove markdown fences and extra formatting from generated code."""
        lines = code.strip().split("\n")
        cleaned: list[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            # Skip markdown code fences
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            cleaned.append(line)

        result = "\n".join(cleaned).strip()

        # Remove leading language identifier if model included it
        lang_markers = {
            "python": ["python", "py"],
            "java": ["java"],
            "cpp": ["cpp", "c++", "c"],
            "csharp": ["csharp", "c#", "cs"],
        }
        markers = lang_markers.get(language, [])
        for marker in markers:
            if result.lower().startswith(marker + "\n"):
                result = result[len(marker) :].strip()
                break

        # Remove trailing explanation sections the model may append
        cut_markers = [
            "\n### Explanation",
            "\n## Explanation",
            "\n**Explanation",
            "\nExplanation:",
            "\nThis code snippet",
            "\nThis code is complete",
            "\nThis program ",
            "\n---\n",
        ]
        for marker in cut_markers:
            idx = result.find(marker)
            if idx > 0:
                result = result[:idx].rstrip()

        return result
