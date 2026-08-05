from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from typing import Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from app.tools.base import BaseTool


class DateTimeParams(BaseModel):
    timezone: str | None = Field(
        default=None,
        description=(
            "IANA timezone name (e.g. 'UTC', 'America/New_York', "
            "'Asia/Kolkata'). Defaults to the host local timezone."
        ),
    )


class DateTimeTool(BaseTool):
    """
    Return current date/time in UTC, local (or requested) zone, and ISO-8601.
    """

    name: ClassVar[str] = "datetime"
    description: ClassVar[str] = (
        "Get the current date and time. Returns UTC time, local (or "
        "requested timezone) time, and ISO-8601 strings."
    )
    parameters_model: ClassVar[type[BaseModel]] = DateTimeParams

    async def execute(self, timezone: str | None = None) -> dict[str, Any]:
        now_utc = datetime.now(dt_timezone.utc)

        if timezone and timezone.strip():
            tz_name = timezone.strip()
            try:
                tz = ZoneInfo(tz_name)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown timezone: {tz_name!r}") from exc
            local = now_utc.astimezone(tz)
            local_label = tz_name
        else:
            local = datetime.now().astimezone()
            local_label = str(local.tzinfo) if local.tzinfo else "local"

        return {
            "utc": {
                "iso": now_utc.isoformat(),
                "date": now_utc.date().isoformat(),
                "time": now_utc.time().replace(microsecond=0).isoformat(),
            },
            "local": {
                "timezone": local_label,
                "iso": local.isoformat(),
                "date": local.date().isoformat(),
                "time": local.time().replace(microsecond=0).isoformat(),
            },
            "iso": now_utc.isoformat(),
        }
