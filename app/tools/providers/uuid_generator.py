from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field

from app.tools.base import BaseTool


class UUIDGeneratorParams(BaseModel):
    count: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of UUID4 values to generate (1–100).",
    )


class UUIDGeneratorTool(BaseTool):
    """Generate one or more RFC 4122 UUID4 values."""

    name: ClassVar[str] = "uuid_generator"
    description: ClassVar[str] = (
        "Generate one or more random UUID4 identifiers."
    )
    parameters_model: ClassVar[type[BaseModel]] = UUIDGeneratorParams

    async def execute(self, count: int = 1) -> dict[str, Any]:
        uuids = [str(uuid4()) for _ in range(count)]
        if count == 1:
            return {"uuid": uuids[0], "uuids": uuids, "count": 1}
        return {"uuids": uuids, "count": count}
