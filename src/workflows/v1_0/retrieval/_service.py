"""Retriever 모듈 관리 — Collection 설정 기반 자동 구성"""

import logging
from functools import lru_cache
from typing import Any, Dict, Optional

from ..config.settings import Collection, CollectionConfig
from ._embeddings import get_embedding
from ._vector_store import VectorStore
from .exceptions import RetrievalConfigError
from .iflow_retriever import IflowRetriever
from .knowhow_retriever import KnowhowRetriever
from .upload_retriever import UploadRetriever

logger = logging.getLogger(__name__)

_collections: Dict[str, CollectionConfig] = {}
_sparse_embeddings: Dict[str, Any] = {}

_RETRIEVER_MAP: Dict[str, type] = {
    Collection.KNOWHOW: KnowhowRetriever,
    Collection.UPLOAD: UploadRetriever,
    Collection.IFLOW: IflowRetriever,
}


def initialize(
    collections: Dict[str, CollectionConfig],
    sparse_embeddings: Optional[Dict[str, Any]] = None,
) -> None:
    """앱 시작 시 1회 호출

    Args:
        sparse_embeddings: hybrid search를 사용할 collection의 sparse encoder 매핑
                           예) {"upload": fitted_bm25} 또는 {"upload": bgem3_ef}
    """
    global _collections, _sparse_embeddings

    if not collections:
        raise RetrievalConfigError("collections는 최소 하나 이상 필요")

    for name in collections:
        if name not in _RETRIEVER_MAP:
            raise RetrievalConfigError(
                f"Collection '{name}'에 매핑된 Retriever가 없습니다. _RETRIEVER_MAP에 등록하세요"
            )

    _collections = dict(collections)
    _sparse_embeddings = dict(sparse_embeddings or {})

    for name in _collections:
        get_retriever(name)

    logger.info(f"Retrieval 초기화 완료 (Retriever: {list(collections.keys())})")


@lru_cache
def get_retriever(name: str) -> Any:
    if name not in _collections:
        raise RetrievalConfigError(
            f"Retriever '{name}'이(가) 등록되지 않았습니다. 사용 가능: {list(_collections.keys())}"
        )

    config = _collections[name]
    vector_store = VectorStore(
        config=config,
        dense_embedding=get_embedding(config.embedding),
        sparse_embedding=_sparse_embeddings.get(name),
        name=name,
    )
    retriever = _RETRIEVER_MAP[name](vector_store)

    try:
        vector_store._ensure_client()
        logger.info(f"VectorStore '{name}' 연결 완료 (collection={config.collection_name})")
    except Exception as e:
        logger.warning(f"VectorStore '{name}' 초기 연결 실패: {e}. 첫 검색 시 재연결 시도")

    return retriever


def knowhow() -> KnowhowRetriever:
    return get_retriever(Collection.KNOWHOW)


def upload() -> UploadRetriever:
    return get_retriever(Collection.UPLOAD)


def iflow() -> IflowRetriever:
    return get_retriever(Collection.IFLOW)


def available_retrievers() -> list:
    return list(_collections.keys())
