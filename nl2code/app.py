"""Streamlit web UI for NL2Code."""

import streamlit as st

from nl2code.engine import CodeGenerator
from nl2code.prompts import LANGUAGE_CONFIGS, SUPPORTED_LANGUAGES


st.set_page_config(
    page_title="NL2Code - Natural Language to Code",
    page_icon="💻",
    layout="wide",
)


@st.cache_resource
def get_generator():
    """Create a cached CodeGenerator instance."""
    return CodeGenerator(backend="auto")


def main():
    st.title("💻 NL2Code")
    st.markdown("**Convert Natural Language to Code** — Python · Java · C++ · C#/.NET")
    st.markdown("---")

    generator = get_generator()

    # Sidebar: Backend status & settings
    with st.sidebar:
        st.header("⚙️ Settings")

        status = generator.get_status()
        st.subheader("Backend Status")

        ollama_status = status["ollama"]
        if ollama_status["available"]:
            st.success("Ollama: Connected")
            if ollama_status["models"]:
                st.caption(f"Models: {', '.join(ollama_status['models'])}")
        else:
            st.warning("Ollama: Not available")
            st.caption("Run `ollama serve` to enable local inference")

        st.info("HuggingFace API: Available (free tier)")
        st.caption(f"Active backend: **{status['active_backend']}**")

        st.markdown("---")

        backend = st.selectbox(
            "Backend",
            options=["auto", "ollama", "huggingface"],
            index=0,
            help="auto = tries Ollama first, falls back to HuggingFace",
        )
        generator.backend = backend

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=2048,
            step=256,
        )

        st.markdown("---")
        st.markdown(
            "**How it works:**\n"
            "1. Describe what you want in plain English\n"
            "2. Pick a target language\n"
            "3. Get working code instantly!\n\n"
            "*Powered by free, open-source AI models*"
        )

    # Main area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Describe Your Code")

        language = st.selectbox(
            "Target Language",
            options=SUPPORTED_LANGUAGES,
            format_func=lambda x: LANGUAGE_CONFIGS[x]["name"],
            index=0,
        )

        task = st.text_area(
            "What should the code do?",
            height=150,
            placeholder="Example: Create a function that takes a list of numbers and returns the top 3 largest numbers sorted in descending order",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_btn = st.button("🚀 Generate Code", type="primary", use_container_width=True)
        with col_btn2:
            explain_btn = st.button("📖 Explain Code", use_container_width=True)

        # Example prompts
        st.markdown("---")
        st.markdown("**💡 Try these examples:**")
        examples = [
            "Sort a list of numbers using bubble sort",
            "Read a CSV file and calculate the average of a column",
            "Implement a binary search tree with insert and search",
            "Create a simple calculator with add, subtract, multiply, divide",
            "Find all prime numbers up to N using Sieve of Eratosthenes",
            "Implement a stack data structure with push, pop, and peek",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{example[:20]}"):
                st.session_state["task_input"] = example
                st.rerun()

    with col2:
        st.subheader(f"📄 Generated {LANGUAGE_CONFIGS[language]['name']} Code")

        if generate_btn and task:
            with st.spinner(f"Generating {LANGUAGE_CONFIGS[language]['name']} code..."):
                result = generator.generate_code(
                    task, language, temperature=temperature, max_tokens=max_tokens
                )

            if result["success"]:
                st.session_state["last_result"] = result
                st.success(f"Generated using **{result['backend_used']}** backend")
                st.code(result["code"], language=language if language != "csharp" else "csharp")

                # Download button
                st.download_button(
                    label=f"📥 Download {result['extension']} file",
                    data=result["code"],
                    file_name=f"generated_code{result['extension']}",
                    mime="text/plain",
                )

                st.caption(f"**Run:** `{result['run_command'].format(filename='generated_code' + result['extension'], classname='Main')}`")
            else:
                st.error(result["error"])

        elif generate_btn and not task:
            st.warning("Please describe what the code should do.")

        # Show explanation
        if explain_btn:
            last = st.session_state.get("last_result")
            if last and last["success"]:
                with st.spinner("Generating explanation..."):
                    explanation = generator.explain_code(last["code"], last["language"])
                if explanation:
                    st.markdown("### 📖 Code Explanation")
                    st.markdown(explanation)
                else:
                    st.warning("Could not generate explanation.")
            else:
                st.warning("Generate code first, then click Explain.")

        # Show last result if exists and no new generation
        if not generate_btn and not explain_btn:
            last = st.session_state.get("last_result")
            if last and last["success"]:
                st.success(f"Generated using **{last['backend_used']}** backend")
                lang = last["language"]
                st.code(last["code"], language=lang if lang != "csharp" else "csharp")


if __name__ == "__main__":
    main()
