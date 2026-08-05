from __future__ import annotations

from app.tools.base import BaseTool
from app.tools.models import ToolSchema
from app.observability.logger import app_logger


class ToolRegistry:
    """
    In-memory registry of available tools.

    Responsibilities
    - Register tools
    - Find tool by name
    - List available tools
    - Return tool schemas for LLM function calling

    Designed so later phases can swap in a plugin loader or MCP
    discovery source without changing ToolService / Gateway.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError("Tool must define a non-empty name")

        name = tool.name.strip().lower()
        if not name:
            raise ValueError("Tool name cannot be blank")

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = tool
        app_logger.info(f"Tool Registered name={name}")

    def unregister(self, name: str) -> None:
        key = name.strip().lower()
        if key in self._tools:
            del self._tools[key]
            app_logger.info(f"Tool Unregistered name={key}")

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name.strip().lower())

    def requires(self, name: str) -> BaseTool:
        tool = self.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(
                f"Tool '{name}' not found. Available: {available}"
            )
        return tool

    def list(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_tools(self) -> list[BaseTool]:
        return [self._tools[name] for name in self.list()]

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema() for tool in self.list_tools()]

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._tools

    def __len__(self) -> int:
        return len(self._tools)
