import os
import yaml
from pathlib import Path
from rich.markdown import Markdown
from rich.console import Console
import argparse

from models_wrapper import LLMManager
from knowledge import KnowledgeIndex  # Use the integrated class

console = Console()


def load_config(path=None):
    if path is None or not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path or "config.yaml")
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--mode", choices=["fast", "deep"])
    return parser.parse_args()


def handle_command(command: str) -> bool:
    """Handles built-in commands like 'exit' or 'help'. Returns True if handled."""
    command = command.strip().lower()
    if command == "exit":
        print("👋 Exiting Stitch.")
        exit(0)
    elif command == "help":
        print("Available commands:\n  exit — Quit Stitch\n  help — Show this message")
        return True
    return False


def load_or_build_index(config):
    """Load the index from disk or build a new one from the vault."""
    vault_dir = Path(os.path.expanduser(config.get("vault_dir", "./vault")))
    embed_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    chunk_size = config.get("chunk_size", 500)
    verbose = True

    ki = KnowledgeIndex(vault_dir, embed_model=embed_model, chunk_size=chunk_size, verbose=verbose)
    ki.build_index()  # build_index automatically checks for changes and loads existing index if possible
    return ki


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.stream:
        config["stream"] = True
    if args.mode:
        config["force_mode"] = args.mode

    print(config.get("welcome_message", "Welcome to Stitch!"))

    try:
        knowledge_index = load_or_build_index(config)
        if knowledge_index.index is None or knowledge_index.index.ntotal == 0:
            print("[Warning] Knowledge index is empty or failed to build. Continuing without context.")
            knowledge_index = None
    except Exception as e:
        print(f"[Warning] Failed to load knowledge index: {e}")
        knowledge_index = None

    llm_manager = LLMManager(config)

    while True:
        try:
            user_input = input("> ")

            if handle_command(user_input):
                continue

            if knowledge_index:
                context_chunks = knowledge_index.query(user_input, top_k=5)
                context = "\n\n".join([c["text"] for c in context_chunks])
                prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            else:
                prompt = f"Question: {user_input}\nAnswer:"

            answer = llm_manager.ask(prompt, stream=config.get("stream", False))
            console.print(Markdown(answer))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Let's try that again...")


if __name__ == "__main__":
    main()
