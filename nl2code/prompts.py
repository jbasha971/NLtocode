"""Prompt templates for each target programming language."""

LANGUAGE_CONFIGS = {
    "python": {
        "name": "Python",
        "extension": ".py",
        "comment": "#",
        "run_command": "python {filename}",
    },
    "java": {
        "name": "Java",
        "extension": ".java",
        "comment": "//",
        "run_command": "javac {filename} && java {classname}",
    },
    "cpp": {
        "name": "C++",
        "extension": ".cpp",
        "comment": "//",
        "run_command": "g++ -o output {filename} && ./output",
    },
    "csharp": {
        "name": "C# (.NET)",
        "extension": ".cs",
        "comment": "//",
        "run_command": "dotnet run",
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())


def build_prompt(task_description: str, language: str) -> str:
    """Build a prompt for the LLM to generate code in the specified language."""
    lang_cfg = LANGUAGE_CONFIGS.get(language)
    if lang_cfg is None:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    lang_name = lang_cfg["name"]

    return f"""You are an expert {lang_name} programmer. Generate clean, well-structured, and working {lang_name} code for the following task.

TASK: {task_description}

RULES:
1. Write ONLY the code. Do NOT include any explanations, markdown formatting, or code fences.
2. Include necessary imports/headers at the top.
3. Add brief comments explaining key logic.
4. The code must be complete and ready to compile/run.
5. Use best practices and proper error handling.
6. If the task involves user input, use standard input/output.

{lang_name} CODE:"""


def build_explanation_prompt(code: str, language: str) -> str:
    """Build a prompt to explain the generated code."""
    lang_name = LANGUAGE_CONFIGS[language]["name"]
    return f"""Explain the following {lang_name} code in simple terms, suitable for a beginner.
Be concise and focus on what each section does.

CODE:
{code}

EXPLANATION:"""
