"""Tests for the NL2Code engine."""

import unittest
from unittest.mock import patch

from nl2code.engine import CodeGenerator
from nl2code.prompts import build_prompt, SUPPORTED_LANGUAGES, LANGUAGE_CONFIGS


class TestPrompts(unittest.TestCase):
    """Test prompt generation."""

    def test_build_prompt_all_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            prompt = build_prompt("sort a list", lang)
            self.assertIn("sort a list", prompt)
            self.assertIn(LANGUAGE_CONFIGS[lang]["name"], prompt)

    def test_build_prompt_invalid_language(self):
        with self.assertRaises(ValueError):
            build_prompt("sort a list", "rust")

    def test_supported_languages(self):
        self.assertIn("python", SUPPORTED_LANGUAGES)
        self.assertIn("java", SUPPORTED_LANGUAGES)
        self.assertIn("cpp", SUPPORTED_LANGUAGES)
        self.assertIn("csharp", SUPPORTED_LANGUAGES)


class TestCodeGenerator(unittest.TestCase):
    """Test the CodeGenerator engine."""

    def setUp(self):
        self.gen = CodeGenerator(backend="auto")

    def test_invalid_language(self):
        result = self.gen.generate_code("sort list", "ruby")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["error"])

    def test_clean_code_removes_fences(self):
        raw = "```python\nprint('hello')\n```"
        cleaned = CodeGenerator._clean_code(raw, "python")
        self.assertNotIn("```", cleaned)
        self.assertIn("print('hello')", cleaned)

    def test_clean_code_removes_lang_prefix(self):
        raw = "python\nprint('hello')"
        cleaned = CodeGenerator._clean_code(raw, "python")
        self.assertEqual(cleaned, "print('hello')")

    def test_status(self):
        status = self.gen.get_status()
        self.assertIn("ollama", status)
        self.assertIn("huggingface", status)
        self.assertIn("supported_languages", status)
        self.assertEqual(len(status["supported_languages"]), 4)

    @patch("nl2code.ollama_backend.is_ollama_available", return_value=False)
    @patch("nl2code.hf_backend.generate", return_value=None)
    def test_no_backend_available(self, mock_hf, mock_ollama):
        result = self.gen.generate_code("sort list", "python")
        self.assertFalse(result["success"])
        self.assertIn("No backend available", result["error"])

    @patch("nl2code.ollama_backend.is_ollama_available", return_value=True)
    @patch("nl2code.ollama_backend.generate", return_value="print('hello world')")
    def test_ollama_generation(self, mock_gen, mock_avail):
        result = self.gen.generate_code("print hello world", "python")
        self.assertTrue(result["success"])
        self.assertEqual(result["backend_used"], "ollama")
        self.assertIn("print", result["code"])

    @patch("nl2code.ollama_backend.is_ollama_available", return_value=False)
    @patch("nl2code.hf_backend.generate", return_value="System.out.println(\"Hello\");")
    def test_hf_fallback(self, mock_hf, mock_ollama):
        result = self.gen.generate_code("print hello", "java")
        self.assertTrue(result["success"])
        self.assertEqual(result["backend_used"], "huggingface")


class TestLanguageConfigs(unittest.TestCase):
    """Test language configuration completeness."""

    def test_all_languages_have_configs(self):
        for lang in SUPPORTED_LANGUAGES:
            cfg = LANGUAGE_CONFIGS[lang]
            self.assertIn("name", cfg)
            self.assertIn("extension", cfg)
            self.assertIn("comment", cfg)
            self.assertIn("run_command", cfg)

    def test_extensions(self):
        self.assertEqual(LANGUAGE_CONFIGS["python"]["extension"], ".py")
        self.assertEqual(LANGUAGE_CONFIGS["java"]["extension"], ".java")
        self.assertEqual(LANGUAGE_CONFIGS["cpp"]["extension"], ".cpp")
        self.assertEqual(LANGUAGE_CONFIGS["csharp"]["extension"], ".cs")


if __name__ == "__main__":
    unittest.main()
