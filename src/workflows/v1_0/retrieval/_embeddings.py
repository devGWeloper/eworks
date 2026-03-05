"""Embedding 인스턴스 관리"""

from functools import lru_cache
from typing import Any

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from ..config.settings import EMBEDDINGS
from .exceptions import RetrievalConfigError


@lru_cache
def get_embedding(name: str) -> Any:
    """name별 Dense Embedding 인스턴스 반환"""
    if name not in EMBEDDINGS:
        raise RetrievalConfigError(
            f"Embedding '{name}'이(가) 등록되지 않았습니다. 사용 가능: {list(EMBEDDINGS.keys())}"
        )
    config = EMBEDDINGS[name]
    return HuggingFaceEndpointEmbeddings(
        model=config.model,
        huggingfacehub_api_token=config.api_key,
    )


@lru_cache
def get_sparse_embedding(language: str = "en") -> Any:
    """BM25 기반 Sparse Embedding 인스턴스 반환

    NOTE: 반환된 인스턴스는 fit()이 호출되지 않은 상태입니다.
    사용 전 반드시 bm25.fit(corpus) 또는 bm25.load("params.json")을 호출하세요.
    컬렉션에 저장된 sparse 벡터와 동일한 encoder·파라미터를 사용해야 합니다.
    """
    from pymilvus.model.sparse import BM25EmbeddingFunction
    from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

    analyzer = build_default_analyzer(language=language)
    return BM25EmbeddingFunction(analyzer)
