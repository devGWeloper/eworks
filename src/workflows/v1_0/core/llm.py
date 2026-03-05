"""LLM 인스턴스 관리"""

import logging
import os
from functools import lru_cache
from typing import Dict, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from ..config.settings import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """LLM 설정 오류 — 모델 미등록 또는 초기화 파라미터 누락"""

    def __init__(self, message: str):
        super().__init__(message)


_model_configs: Dict[str, LLMModelConfig] = {}
_default_model_name: str = ""


def initialize(models: Dict[str, LLMModelConfig], default_model: str) -> None:
    """앱 시작 시 1회 호출 — 모델 설정 등록 + 기본 모델 사전 생성"""
    global _model_configs, _default_model_name

    if not models:
        raise LLMConfigError("models는 비어 있을 수 없습니다")
    if default_model not in models:
        raise LLMConfigError(f"default_model '{default_model}'이 models에 없습니다")

    _model_configs = dict(models)
    _default_model_name = default_model
    get_llm(default_model)  # 기본 모델 사전 생성

    logger.info(f"LLM 초기화 완료: {list(models.keys())}, default={default_model}")


def get_llm(model_name_key: Optional[str] = None) -> BaseChatModel:
    """이름 정규화 → 캐시 위임, None이면 기본 모델 반환"""
    return _get_cached_model(model_name_key or _default_model_name)


@lru_cache(maxsize=None)
def _get_cached_model(name: str) -> BaseChatModel:
    """모델 이름별 인스턴스 캐싱"""
    if name not in _model_configs:
        raise LLMConfigError(f"등록되지 않은 모델: '{name}'")

    config = _model_configs[name]
    logger.info(f"LLM 모델 생성: {name} ({config.model})")
    return _create_model(config)


def _create_model(config: LLMModelConfig) -> BaseChatModel:
    """LLMModelConfig → ChatOpenAI 인스턴스 생성"""
    kwargs = dict(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
    )

    if os.getenv("SSL_VERIFY", "true").lower() == "false":
        kwargs["http_client"] = httpx.Client(verify=False)
        kwargs["http_async_client"] = httpx.AsyncClient(verify=False)

    return ChatOpenAI(**kwargs)


async def call_llm(
    messages: list,
    tools: list | None = None,
    model_name_key: Optional[str] = None,
) -> BaseMessage:
    """LLM 호출 — messages + optional tools"""
    llm = get_llm(model_name_key)
    if tools:
        llm = llm.bind_tools(tools)
    return await llm.ainvoke(messages)
