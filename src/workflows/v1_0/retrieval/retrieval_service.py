"""RetrievalService — 제네릭 접근자 [Preset] + 도메인 검색 메서드 [Domain] 확장 포인트"""

from core.services.retrieval.base_collection import BaseCollection
from core.services.retrieval.retrieval_service_core import RetrievalServiceCore

from ..config.settings import CollectionName


class RetrievalService(RetrievalServiceCore):
    """제네릭 접근자 [Preset] + 도메인 검색 메서드 [Domain] 확장 포인트"""

    # ── Collection 접근자 [Preset] ──

    @property
    def knowhow(self) -> BaseCollection:
        """knowhow Collection 접근자"""
        return self.get_collection(CollectionName.KNOWHOW)

    # ── 도메인 검색 메서드 [Domain] ──

    async def search_event(self, event_name: str) -> list[tuple]:
        """이벤트 검색"""
        return await self.knowhow.similarity_search_with_score(
            query=f"이벤트: {event_name}", k=5
        )
