from functools import lru_cache

from app.gateway.gateway import Gateway
from app.memory.service import MemoryService


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService()


@lru_cache
def get_gateway() -> Gateway:
    return Gateway(memory=get_memory_service())
