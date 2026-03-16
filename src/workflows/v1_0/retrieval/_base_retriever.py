"""BaseRetriever — VectorStore 검색 위임 기본 클래스"""

from typing import Optional

from langchain_core.documents import Document

from ._vector_store import VectorStore


class BaseRetriever:
    _categories: dict = {}

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    async def similarity_search(
        self, query: str, k: int = 5, expr: Optional[str] = None
    ) -> list[Document]:
        return await self._vector_store.similarity_search(query=query, k=k, expr=expr)

    async def similarity_search_with_score(
        self, query: str, k: int = 5, expr: Optional[str] = None
    ) -> list[tuple[Document, float]]:
        return await self._vector_store.similarity_search_with_score(query=query, k=k, expr=expr)

    async def hybrid_search_with_score(
        self,
        query: str,
        k: int = 5,
        expr: Optional[str] = None,
        sparse_weight: float = 0.1,
        dense_weight: float = 0.9,
    ) -> list[tuple[Document, float]]:
        return await self._vector_store.hybrid_search_with_score(
            query=query, k=k, expr=expr, sparse_weight=sparse_weight, dense_weight=dense_weight,
        )

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        expr: Optional[str] = None,
        sparse_weight: float = 0.1,
        dense_weight: float = 0.9,
    ) -> list[Document]:
        return [doc for doc, _ in await self.hybrid_search_with_score(query, k, expr, sparse_weight, dense_weight)]
