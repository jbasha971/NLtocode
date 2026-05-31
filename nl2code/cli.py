"""Command-line interface for NL2Code."""

import argparse
import sys

from nl2code.engine import CodeGenerator
from nl2code.prompts import SUPPORTED_LANGUAGES, LANGUAGE_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        prog="nl2code",
        description="Convert natural language descriptions to code in Python, Java, C++, or C#/.NET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nl2code -t "sort a list of numbers" -l python
  nl2code -t "read a file and count words" -l java
  nl2code -t "implement binary search" -l cpp
  nl2code -t "create a REST API endpoint" -l csharp
  nl2code --interactive
        """,
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        help="Natural language description of the code to generate",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Target programming language (default: python)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "-b", "--backend",
        type=str,
        choices=["auto", "ollama", "huggingface"],
        default="auto",
        help="Backend to use (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (depends on backend)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Also generate an explanation of the code",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show backend status and exit",
    )

    args = parser.parse_args()

    generator = CodeGenerator(backend=args.backend)

    if args.status:
        _show_status(generator)
        return

    if args.interactive:
        _interactive_mode(generator)
        return

    if not args.task:
        parser.error("Please provide a task with -t/--task, or use --interactive mode")

    result = generator.generate_code(args.task, args.language)

    if not result["success"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    code = result["code"]

    if args.output:
        with open(args.output, "w") as f:
            f.write(code + "\n")
        print(f"Code saved to: {args.output}")
        print(f"Run with: {result['run_command'].format(filename=args.output, classname='Main')}")
    else:
        print(f"\n{'='*60}")
        print(f" Generated {result['language_name']} Code")
        print(f" Backend: {result['backend_used']}")
        print(f"{'='*60}\n")
        print(code)
        print(f"\n{'='*60}")

    if args.explain:
        print("\n📝 Explanation:\n")
        explanation = generator.explain_code(code, args.language)
        if explanation:
            print(explanation)
        else:
            print("(Could not generate explanation)")


def _show_status(generator: CodeGenerator):
    """Display backend status."""
    status = generator.get_status()
    print("\n=== NL2Code Backend Status ===\n")

    ollama = status["ollama"]
    print(f"Ollama: {'Available' if ollama['available'] else 'Not available'}")
    if ollama["available"] and ollama["models"]:
        print(f"  Models: {', '.join(ollama['models'])}")

    print(f"HuggingFace API: Available (free tier)")
    print(f"\nActive backend: {status['active_backend']}")
    print(f"Supported languages: {', '.join(status['supported_languages'])}")
    print()


def _interactive_mode(generator: CodeGenerator):
    """Run interactive session."""
    print("\n" + "=" * 60)
    print("  NL2Code - Natural Language to Code Converter")
    print("  Type 'quit' to exit, 'status' for backend info")
    print("=" * 60)

    lang_options = {str(i + 1): lang for i, lang in enumerate(SUPPORTED_LANGUAGES)}

    while True:
        print("\nAvailable languages:")
        for num, lang in lang_options.items():
            print(f"  {num}. {LANGUAGE_CONFIGS[lang]['name']}")

        lang_choice = input("\nSelect language [1-4] (default: 1): ").strip() or "1"

        if lang_choice.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if lang_choice.lower() == "status":
            _show_status(generator)
            continue

        language = lang_options.get(lang_choice)
        if not language:
            print("Invalid choice. Please select 1-4.")
            continue

        task = input(f"\nDescribe what the {LANGUAGE_CONFIGS[language]['name']} code should do:\n> ").strip()

        if not task:
            print("Please provide a description.")
            continue

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print(f"\nGenerating {LANGUAGE_CONFIGS[language]['name']} code...")
        result = generator.generate_code(task, language)

        if result["success"]:
            print(f"\n{'='*60}")
            print(f" Generated {result['language_name']} Code")
            print(f" Backend: {result['backend_used']}")
            print(f"{'='*60}\n")
            print(result["code"])
            print(f"\n{'='*60}")

            save = input("\nSave to file? (y/N): ").strip().lower()
            if save == "y":
                default_name = f"output{result['extension']}"
                filename = input(f"Filename [{default_name}]: ").strip() or default_name
                with open(filename, "w") as f:
                    f.write(result["code"] + "\n")
                print(f"Saved to: {filename}")
        else:
            print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()
