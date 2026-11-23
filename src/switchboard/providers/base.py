"""Provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..types import ChatRequest, ChatResponse, Chunk


class Provider(ABC):
    name: str = "provider"

    @abstractmethod
    async def chat(self, req: ChatRequest, upstream_model: str) -> ChatResponse: ...

    @abstractmethod
    def stream(self, req: ChatRequest, upstream_model: str) -> AsyncIterator[Chunk]: ...

    async def health(self) -> bool:
        return True

    def models(self) -> list[str]:
        return []
