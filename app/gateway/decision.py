from dataclasses import dataclass

from app.config.constants import Provider, RouteType


@dataclass
class Decision:
    provider: Provider
    route: RouteType
    model: str