from app.prompts.registry import PROMPTS, versions
from app.prompts.versioning import PromptVersion, PromptVersionError


class PromptManager:
    """
    Loads and renders versioned prompt templates.
    """

    def __init__(self) -> None:
        self.versions = versions

    def load(
        self,
        name: str,
        version: int | None = None,
    ) -> str:
        self._ensure_known(name)
        return self.versions.get(name, version).read()

    def render(
        self,
        name: str,
        version: int | None = None,
        **variables,
    ) -> str:
        prompt = self.load(name, version=version)

        for key, value in variables.items():
            prompt = prompt.replace(
                f"{{{{{key}}}}}",
                str(value),
            )

        return prompt

    def get_version(
        self,
        name: str,
        version: int | None = None,
    ) -> PromptVersion:
        self._ensure_known(name)
        return self.versions.get(name, version)

    def list_versions(self, name: str) -> list[int]:
        self._ensure_known(name)
        return self.versions.list_versions(name)

    def activate(self, name: str, version: int) -> PromptVersion:
        """Activate a version for experimentation."""
        self._ensure_known(name)
        return self.versions.set_active(name, version)

    def rollback(self, name: str, version: int) -> PromptVersion:
        """Roll back the active prompt to a previous version."""
        self._ensure_known(name)
        return self.versions.rollback(name, version)

    def register_version(
        self,
        name: str,
        content: str,
        *,
        activate: bool = False,
    ) -> PromptVersion:
        """Create a new prompt version (optionally activate it)."""
        self._ensure_known(name)
        return self.versions.register_version(
            name,
            content,
            activate=activate,
        )

    def _ensure_known(self, name: str) -> None:
        if name not in PROMPTS:
            raise PromptVersionError(f"Unknown prompt '{name}'")
