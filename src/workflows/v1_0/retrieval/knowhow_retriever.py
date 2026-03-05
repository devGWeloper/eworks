"""KnowhowRetriever — knowhow 컬렉션 도메인 검색"""

from langchain_core.documents import Document

from ._base_retriever import BaseRetriever


class KnowhowRetriever(BaseRetriever):
    """knowhow 컬렉션 Retriever — 도메인 검색 메서드 정의"""

    _categories = {
        "event": "이벤트 명세, 이벤트 발생 조건, 이벤트 처리 방법",
        "log": "로그 내용 설명, 로그 분석 방법, 로그 항목 의미",
    }

    # ── 도메인 검색 메서드 ──

    async def search_event(self, event_name: str, k: int = 5) -> list[tuple[Document, float]]:
        """이벤트 명세 검색"""
        return await self.similarity_search_with_score(query=f"이벤트: {event_name}", k=k, expr='category == "event"')
