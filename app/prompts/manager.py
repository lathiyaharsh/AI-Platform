from pathlib import Path

from app.prompts.registry import PROMPTS


class PromptManager:
    """
    Loads and renders prompt templates.
    """

    def load(self, name: str) -> str:
        path: Path = PROMPTS[name]

        return path.read_text(encoding="utf-8")

    def render(
        self,
        name: str,
        **variables,
    ) -> str:

        prompt = self.load(name)

        for key, value in variables.items():
            prompt = prompt.replace(
                f"{{{{{key}}}}}",
                str(value),
            )

        return prompt