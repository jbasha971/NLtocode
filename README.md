# NL2Code 💻

**Convert Natural Language to Code** — A college intern project that generates working code in **Python, Java, C++, and C#/.NET** from plain English descriptions.

> **Zero cost** — runs entirely on free, open-source AI models. No paid API tokens required.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **4 Target Languages**: Python, Java, C++, C# (.NET)
- **Free AI Backends**:
  - **Ollama** (primary) — runs locally, completely offline, no tokens needed
  - **HuggingFace Inference API** (fallback) — free tier, no API key required
- **Two Interfaces**:
  - 🌐 **Web UI** — built with Streamlit, easy to use
  - ⌨️ **CLI** — for terminal lovers and scripting
- **Code Explanation** — get plain-English explanations of generated code
- **Auto Backend Selection** — tries Ollama first, falls back to HuggingFace automatically

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/jbasha971/NLtocode.git
cd NLtocode

# Install Python dependencies
pip install -r requirements.txt

# Install the package (enables `nl2code` CLI command)
pip install -e .
```

### 2. Install Ollama (Recommended)

Ollama runs AI models locally for free. Install it from [ollama.com/download](https://ollama.com/download):

- **Windows**: Download and run the installer from [ollama.com/download](https://ollama.com/download). Ollama runs as a background service automatically.
- **Linux/macOS**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

After installing, pull the code generation model (~1 GB, one-time download):
```bash
ollama pull qwen2.5-coder:1.5b
```

> **Note**: If you skip Ollama, the app will automatically use HuggingFace's free API (requires internet).

### 3. Run the App

**Web UI (recommended):**
```bash
python -m streamlit run nl2code/app.py
```
Opens at `http://localhost:8501`

**CLI:**
```bash
# One-shot mode
nl2code -t "sort a list of numbers using bubble sort" -l python

# Interactive mode
nl2code --interactive

# Save to file
nl2code -t "implement binary search" -l java -o BinarySearch.java

# Check backend status
nl2code --status
```

---

## Running in VS Code (Step-by-Step)

### Prerequisites

1. Install **Python 3.9+** from [python.org](https://www.python.org/downloads/)
2. Install **VS Code** from [code.visualstudio.com](https://code.visualstudio.com/)
3. Install the **Python extension** in VS Code (Extensions tab → search "Python" → install by Microsoft)
4. Install **Ollama** from [ollama.com/download](https://ollama.com/download)

### Setup

1. Clone/download this repo and open the folder in VS Code (**File → Open Folder**)
2. Open the terminal in VS Code: **Terminal → New Terminal** (or press `` Ctrl+` ``)
3. Run these commands:

```powershell
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Pull the AI model (one-time, ~1 GB)
ollama pull qwen2.5-coder:1.5b
```

### Run the Web UI

```powershell
python -m streamlit run nl2code/app.py
```

This opens the web app at `http://localhost:8501` in your browser.

### Run the CLI

```powershell
python -m nl2code.cli -t "sort a list of numbers" -l python
python -m nl2code.cli --interactive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ollama` not recognized | Restart VS Code after installing Ollama |
| `streamlit` not recognized | Use `python -m streamlit run nl2code/app.py` instead |
| `pip` not recognized | Use `python -m pip install -r requirements.txt` |
| "bind: address already in use" on `ollama serve` | Ollama is already running in the background (Windows does this automatically). Just skip `ollama serve` |
| Slow first code generation | Normal on CPU — first request takes 10-30s, subsequent ones are faster |

---

## Project Structure

```
nl2code/
├── nl2code/
│   ├── __init__.py           # Package init
│   ├── app.py                # Streamlit web UI
│   ├── cli.py                # Command-line interface
│   ├── engine.py             # Core code generation engine
│   ├── ollama_backend.py     # Ollama local LLM backend
│   ├── hf_backend.py         # HuggingFace API backend
│   └── prompts.py            # Prompt templates per language
├── tests/
│   └── test_engine.py        # Unit tests
├── examples/
│   └── sample_queries.txt    # Example queries to try
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

---

## Usage Examples

### Python
```
Task: "Create a function that finds all prime numbers up to N using the Sieve of Eratosthenes"
```

### Java
```
Task: "Implement a student grade management system with add, remove, and average calculation"
```

### C++
```
Task: "Implement a doubly linked list with insert, delete, and display operations"
```

### C# (.NET)
```
Task: "Create a simple REST API controller for managing a to-do list"
```

---

## How It Works

1. **User Input** → You describe what the code should do in plain English
2. **Prompt Engineering** → The description is wrapped in a language-specific prompt template
3. **AI Generation** → The prompt is sent to a code-specialized LLM:
   - **Ollama** (local, offline) → Uses `qwen2.5-coder:1.5b` model
   - **HuggingFace** (online, free) → Uses `bigcode/starcoder2-3b` model
4. **Post-Processing** → Generated code is cleaned (remove markdown fences, etc.)
5. **Output** → Clean, ready-to-run code is displayed with run instructions

---

## Configuration

| Option | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| Backend | `--backend` | `auto` | `auto`, `ollama`, or `huggingface` |
| Language | `-l` | `python` | `python`, `java`, `cpp`, `csharp` |
| Output File | `-o` | stdout | Save code to a file |
| Explain | `--explain` | off | Generate code explanation |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Tech Stack

- **Python 3.9+** — main application language
- **Streamlit** — web UI framework
- **Ollama** — local LLM runtime (uses `qwen2.5-coder:1.5b`)
- **HuggingFace Inference API** — free cloud fallback
- **Requests** — HTTP client for API calls

---

## License

MIT License — free for academic and personal use.

---

*Built as a college intern project — demonstrating AI-powered code generation without paid API tokens.*
