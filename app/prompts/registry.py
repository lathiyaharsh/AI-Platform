from pathlib import Path

from app.prompts.versioning import PromptVersionStore

PROMPT_DIR = Path(__file__).parent / "templates"

PROMPTS = (
    "chat",
    "code",
    "summarize",
)

versions = PromptVersionStore(PROMPT_DIR)
