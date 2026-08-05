from __future__ import annotations

from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field, model_validator

from app.tools.base import BaseTool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherParams(BaseModel):
    """Locate weather by city name and/or explicit coordinates."""

    city: str | None = Field(
        default=None,
        description="City name to geocode (e.g. 'London', 'Tokyo')",
    )
    latitude: float | None = Field(
        default=None,
        description="Latitude in decimal degrees",
    )
    longitude: float | None = Field(
        default=None,
        description="Longitude in decimal degrees",
    )

    @model_validator(mode="after")
    def require_location(self) -> WeatherParams:
        has_city = bool(self.city and self.city.strip())
        has_coords = self.latitude is not None and self.longitude is not None
        if not has_city and not has_coords:
            raise ValueError(
                "Provide city or both latitude and longitude"
            )
        return self


class WeatherTool(BaseTool):
    """
    Current weather via the free Open-Meteo API.

    No API key required. Supports city geocoding or explicit lat/lon.
    """

    name: ClassVar[str] = "weather"
    description: ClassVar[str] = (
        "Get current weather conditions for a city or coordinates "
        "using the Open-Meteo API. Returns temperature (°C), "
        "wind speed, weather code, and location metadata."
    )
    parameters_model: ClassVar[type[BaseModel]] = WeatherParams

    def __init__(self, *, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        city: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        lat = latitude
        lon = longitude
        resolved_name = city.strip() if city else None
        country: str | None = None

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            if lat is None or lon is None:
                assert resolved_name is not None
                geo = await self._geocode(client, resolved_name)
                lat = geo["latitude"]
                lon = geo["longitude"]
                resolved_name = geo["name"]
                country = geo.get("country")

            forecast = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "timezone": "auto",
                },
            )
            forecast.raise_for_status()
            data = forecast.json()

        current = data.get("current_weather") or {}
        return {
            "location": {
                "name": resolved_name,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "timezone": data.get("timezone"),
            },
            "current": {
                "temperature_c": current.get("temperature"),
                "windspeed_kmh": current.get("windspeed"),
                "winddirection_deg": current.get("winddirection"),
                "weathercode": current.get("weathercode"),
                "time": current.get("time"),
                "is_day": current.get("is_day"),
            },
            "source": "open-meteo",
        }

    @staticmethod
    async def _geocode(
        client: httpx.AsyncClient,
        city: str,
    ) -> dict[str, Any]:
        response = await client.get(
            _GEOCODE_URL,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"City not found: {city!r}")
        place = results[0]
        return {
            "name": place.get("name", city),
            "country": place.get("country"),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        }
