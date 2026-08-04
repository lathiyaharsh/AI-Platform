from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

VERSION_PATTERN = re.compile(r"^v(\d+)\.txt$")


class PromptVersionError(Exception):
    """Raised when a prompt version cannot be resolved."""


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: int
    path: Path

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")


class PromptVersionStore:
    """
    File-based prompt version history.

    Layout:
        templates/{name}/v1.txt
        templates/{name}/v2.txt
        templates/active.json   # {"chat": 2, ...}
    """

    def __init__(self, templates_dir: Path) -> None:
        self.templates_dir = templates_dir
        self.active_path = templates_dir / "active.json"
        self._lock = Lock()
        self._active: dict[str, int] = self._load_active()

    def get(
        self,
        name: str,
        version: int | None = None,
    ) -> PromptVersion:
        resolved = version if version is not None else self.get_active(name)
        path = self._version_path(name, resolved)

        if not path.exists():
            raise PromptVersionError(
                f"Prompt '{name}' version v{resolved} not found"
            )

        return PromptVersion(name=name, version=resolved, path=path)

    def list_versions(self, name: str) -> list[int]:
        prompt_dir = self._prompt_dir(name)

        if not prompt_dir.is_dir():
            raise PromptVersionError(f"Unknown prompt '{name}'")

        versions: list[int] = []

        for path in prompt_dir.iterdir():
            match = VERSION_PATTERN.match(path.name)
            if match and path.is_file():
                versions.append(int(match.group(1)))

        return sorted(versions)

    def latest(self, name: str) -> int:
        versions = self.list_versions(name)

        if not versions:
            raise PromptVersionError(f"No versions found for prompt '{name}'")

        return versions[-1]

    def get_active(self, name: str) -> int:
        with self._lock:
            if name in self._active:
                return self._active[name]

        return self.latest(name)

    def set_active(self, name: str, version: int) -> PromptVersion:
        """
        Activate a specific version (rollback or experiment).
        """
        prompt = self.get(name, version)

        with self._lock:
            self._active[name] = version
            self._save_active()

        return prompt

    def rollback(self, name: str, version: int) -> PromptVersion:
        return self.set_active(name, version)

    def register_version(
        self,
        name: str,
        content: str,
        *,
        activate: bool = False,
    ) -> PromptVersion:
        """
        Write a new version file (vN+1) for a prompt.
        """
        prompt_dir = self._prompt_dir(name)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        try:
            next_version = self.latest(name) + 1
        except PromptVersionError:
            next_version = 1

        path = self._version_path(name, next_version)
        path.write_text(content, encoding="utf-8")

        prompt = PromptVersion(name=name, version=next_version, path=path)

        if activate:
            self.set_active(name, next_version)

        return prompt

    def _prompt_dir(self, name: str) -> Path:
        return self.templates_dir / name

    def _version_path(self, name: str, version: int) -> Path:
        return self._prompt_dir(name) / f"v{version}.txt"

    def _load_active(self) -> dict[str, int]:
        if not self.active_path.exists():
            return {}

        data = json.loads(self.active_path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}

    def _save_active(self) -> None:
        self.active_path.write_text(
            json.dumps(self._active, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
