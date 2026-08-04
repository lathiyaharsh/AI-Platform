from functools import lru_cache

from app.gateway.gateway import Gateway


@lru_cache
def get_gateway() -> Gateway:
    return Gateway()