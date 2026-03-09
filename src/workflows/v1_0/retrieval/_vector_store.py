"""VectorStore — pymilvus MilvusClient 기반 벡터 DB 검색 + 자동 재연결"""

import asyncio
import logging
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from ..config.settings import CollectionConfig
from .exceptions import RetrievalConnectionError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2]


def _parse_hit(hit) -> tuple[dict, float]:
    if isinstance(hit, dict):
        return hit.get("entity", {}), float(hit.get("distance", 0.0))
    entity = getattr(hit, "entity", None) or getattr(hit, "fields", {})
    return entity, float(getattr(hit, "distance", 0.0))


class VectorStore:
    def __init__(
        self,
        config: CollectionConfig,
        dense_embedding: Any,
        sparse_embedding: Any = None,
        name: str = "",
    ):
        self._config = config
        self._dense_embedding = dense_embedding
        self._sparse_embedding = sparse_embedding
        self._name = name
        self._client: Optional[MilvusClient] = None

    def _ensure_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._config.uri, token=self._config.token)
            logger.info(f"VectorStore '{self._name}' 연결 완료")
        return self._client

    async def _execute_with_retry(self, operation):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait = BACKOFF_SECONDS[attempt - 1]
                    logger.warning(f"VectorStore '{self._name}' 재시도 {attempt + 1}/{MAX_RETRIES} ({wait}s 대기)")
                    await asyncio.sleep(wait)
                    self._client = MilvusClient(uri=self._config.uri, token=self._config.token)
                return await operation()
            except Exception as e:
                last_error = e
                logger.warning(f"VectorStore '{self._name}' 실패 (시도 {attempt + 1}/{MAX_RETRIES}): {e}")
        raise RetrievalConnectionError(self._name, last_error)

    async def _embed(self, query: str) -> list[float]:
        return await asyncio.get_event_loop().run_in_executor(None, self._dense_embedding.embed_query, query)

    async def _embed_sparse(self, query: str):
        if self._sparse_embedding is None:
            raise ValueError(f"'{self._name}': sparse_embedding 미설정")
        result = await asyncio.get_event_loop().run_in_executor(None, self._sparse_embedding.encode_queries, [query])
        return result[0]

    def _parse_hits(self, hits: list) -> list[tuple[Document, float]]:
        results = []
        for hit in hits:
            entity, score = _parse_hit(hit)
            text = entity.get(self._config.text_field, "")
            metadata = {k: v for k, v in entity.items() if k != self._config.text_field}
            results.append((Document(page_content=text, metadata=metadata), score))
        return results

    # ── 검색 ──

    async def similarity_search_with_score(
        self, query: str, k: int = 5, expr: Optional[str] = None
    ) -> list[tuple[Document, float]]:
        async def _op():
            results = self._ensure_client().search(
                collection_name=self._config.collection_name,
                data=[await self._embed(query)],
                anns_field=self._config.vector_field,
                search_params=self._config.dense_search_params or {"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=k,
                filter=expr,
                output_fields=["*"],
            )
            return self._parse_hits(results[0])

        return await self._execute_with_retry(_op)

    async def similarity_search(
        self, query: str, k: int = 5, expr: Optional[str] = None
    ) -> list[Document]:
        return [doc for doc, _ in await self.similarity_search_with_score(query, k, expr)]

    async def hybrid_search_with_score(
        self,
        query: str,
        k: int = 5,
        expr: Optional[str] = None,
        ranker: Literal["rrf", "weighted"] = "rrf",
        sparse_weight: float = 0.2,
    ) -> list[tuple[Document, float]]:
        # vector_fields 또는 sparse_vector_field 중 하나는 있어야 hybrid
        has_multi_dense = self._config.vector_fields and len(self._config.vector_fields) > 1
        has_sparse = bool(self._config.sparse_vector_field)
        if not has_multi_dense and not has_sparse:
            raise ValueError(f"'{self._name}': hybrid search 불가 — vector_fields 또는 sparse_vector_field 설정 필요")

        async def _op():
            dense_vec = await self._embed(query)
            fields_weights = self._config.vector_fields or {self._config.vector_field: 1.0}

            # sparse 먼저, dense 뒤에 — WeightedRanker 순서 일치
            reqs, weights = [], []
            if has_sparse:
                sparse_vec = await self._embed_sparse(query)
                reqs.append(AnnSearchRequest(
                    data=[sparse_vec], anns_field=self._config.sparse_vector_field,
                    param={"metric_type": self._config.sparse_metric_type, "params": {}},
                    limit=k, expr=expr,
                ))
                weights.append(sparse_weight)

            for field_name, weight in fields_weights.items():
                reqs.append(AnnSearchRequest(
                    data=[dense_vec], anns_field=field_name,
                    param=self._config.dense_search_params or {"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=k, expr=expr,
                ))
                weights.append(weight)

            reranker = WeightedRanker(*weights) if ranker == "weighted" else RRFRanker(k=60)
            results = self._ensure_client().hybrid_search(
                collection_name=self._config.collection_name,
                reqs=reqs,
                ranker=reranker,
                limit=k,
                output_fields=["*"],
            )
            return self._parse_hits(results[0])

        return await self._execute_with_retry(_op)

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        expr: Optional[str] = None,
        ranker: Literal["rrf", "weighted"] = "rrf",
        sparse_weight: float = 0.2,
    ) -> list[Document]:
        return [doc for doc, _ in await self.hybrid_search_with_score(query, k, expr, ranker, sparse_weight)]
