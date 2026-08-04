from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "templates"

PROMPTS = {
    "chat": PROMPT_DIR / "chat.txt",
    "code": PROMPT_DIR / "code.txt",
    "summarize": PROMPT_DIR / "summarize.txt",
}