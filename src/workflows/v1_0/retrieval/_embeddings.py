"""Embedding 인스턴스 관리"""

from functools import lru_cache
from typing import Any

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from ..config.settings import EMBEDDINGS
from .exceptions import RetrievalConfigError


@lru_cache
def get_embedding(name: str) -> Any:
    if name not in EMBEDDINGS:
        raise RetrievalConfigError(
            f"Embedding '{name}'이(가) 등록되지 않았습니다. 사용 가능: {list(EMBEDDINGS.keys())}"
        )
    config = EMBEDDINGS[name]
    return HuggingFaceEndpointEmbeddings(
        model=config.model,
        huggingfacehub_api_token=config.api_key,
    )
