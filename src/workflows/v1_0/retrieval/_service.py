"""Retriever 모듈 관리 — Collection 설정 기반 자동 구성"""

import logging
from functools import lru_cache
from typing import Any, Dict

from ..config.settings import Collection, CollectionConfig
from ._embeddings import get_embedding, get_sparse_embedding
from ._vector_store import VectorStore
from .exceptions import RetrievalConfigError
from .knowhow_retriever import KnowhowRetriever
from .upload_retriever import UploadRetriever

logger = logging.getLogger(__name__)

_collections: Dict[str, CollectionConfig] = {}

# ── Collection → Retriever 매핑 ──

_RETRIEVER_MAP: Dict[str, type] = {
    Collection.KNOWHOW: KnowhowRetriever,
    Collection.UPLOAD: UploadRetriever,
}


# ── 초기화 ──


def initialize(collections: Dict[str, CollectionConfig]) -> None:
    """앱 시작 시 1회 호출 — config 검증/등록 + warmup"""
    global _collections

    if not collections:
        raise RetrievalConfigError("collections는 최소 하나 이상 필요")

    for name in collections:
        if name not in _RETRIEVER_MAP:
            raise RetrievalConfigError(
                f"Collection '{name}'에 매핑된 Retriever가 없습니다. _RETRIEVER_MAP에 등록하세요"
            )

    _collections = dict(collections)

    for name in _collections:
        get_retriever(name)

    logger.info(f"Retrieval 초기화 완료 (Retriever: {list(collections.keys())})")


# ── Retriever 접근자 ──


@lru_cache
def get_retriever(name: str) -> Any:
    """name별 Retriever 인스턴스 반환"""
    if name not in _collections:
        raise RetrievalConfigError(
            f"Retriever '{name}'이(가) 등록되지 않았습니다. 사용 가능: {list(_collections.keys())}"
        )

    config = _collections[name]
    dense_embedding = get_embedding(config.embedding)
    sparse_embedding = get_sparse_embedding() if config.sparse_vector_field else None

    vector_store = VectorStore(
        config=config,
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        name=name,
    )
    retriever = _RETRIEVER_MAP[name](vector_store)

    # warmup 시도 — 실패해도 앱 기동 차단 안함
    try:
        vector_store._ensure_client()
        logger.info(f"VectorStore '{name}' 연결 완료 (collection={config.collection_name})")
    except Exception as e:
        logger.warning(
            f"VectorStore '{name}' 초기 연결 실패 (collection={config.collection_name}): {e}. 첫 검색 시 재연결 시도"
        )

    return retriever


def knowhow() -> KnowhowRetriever:
    """knowhow Retriever 반환"""
    return get_retriever(Collection.KNOWHOW)


def upload() -> UploadRetriever:
    """upload Retriever 반환"""
    return get_retriever(Collection.UPLOAD)


def available_retrievers() -> list:
    """등록된 Retriever 이름 목록"""
    return list(_collections.keys())
