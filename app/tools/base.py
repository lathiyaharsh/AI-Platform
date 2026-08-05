from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from app.tools.models import ToolSchema


class EmptyToolParams(BaseModel):
    """Default parameter model for tools that take no arguments."""


class BaseTool(ABC):
    """
    Abstract base for every platform tool.

    Subclasses declare:
    - name
    - description
    - parameters_model (Pydantic) → JSON schema for LLMs
    - async execute(**kwargs)

    New tools (REST, SQL, Slack, MCP, …) only subclass BaseTool
    and register with ToolRegistry — Gateway stays unchanged.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters_model: ClassVar[type[BaseModel]] = EmptyToolParams

    def schema(self) -> ToolSchema:
        """Return the JSON schema used for LLM function calling."""
        json_schema = self.parameters_model.model_json_schema()
        # Strip Pydantic metadata noise; keep LLM-friendly shape.
        parameters = {
            "type": json_schema.get("type", "object"),
            "properties": json_schema.get("properties", {}),
        }
        if "required" in json_schema:
            parameters["required"] = json_schema["required"]
        if "$defs" in json_schema:
            parameters["$defs"] = json_schema["$defs"]
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        """
        Validate and coerce raw LLM arguments.

        Raises ValidationError on invalid input (caught by ToolExecutor).
        """
        return self.parameters_model.model_validate(arguments or {})

    async def run(self, arguments: dict[str, Any] | None = None) -> Any:
        """Validate arguments then execute."""
        validated = self.validate_arguments(arguments or {})
        return await self.execute(**validated.model_dump())

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Perform the tool action. Must be implemented by subclasses."""
        raise NotImplementedError
