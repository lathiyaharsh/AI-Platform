from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> list[float]:
        pass