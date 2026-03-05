"""BaseAgent — Agent 추상 기본 클래스"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..state import GraphState
from ._executor import AgentExecutor

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent 추상 기본 클래스 — 필수 속성 검증 + AgentExecutor 자동 생성"""

    # ── 필수 속성 검증 ──
    _REQUIRED_ATTRS: frozenset = frozenset({"system_prompt", "user_prompt_template"})

    # ── LLM 설정 (서브클래스에서 오버라이드) ──
    system_prompt: str = ""
    user_prompt_template: str = ""
    model_name_key: Optional[str] = None

    # ── Intent 메타데이터 ──
    intent_id: str = ""
    intent_name: str = ""
    intent_description: str = ""
    intent_examples: list = []
    intent_parameters: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # 중간 추상 클래스는 검증 스킵
        for attr in cls._REQUIRED_ATTRS:
            if not getattr(cls, attr, ""):
                raise AttributeError(f"{cls.__name__}: 필수 속성 '{attr}'이 설정되지 않았습니다")

    def __init__(self):
        self._executor = AgentExecutor(
            system_prompt=self.system_prompt or None,
            user_prompt_template=self.user_prompt_template or None,
            model_name_key=self.model_name_key,
            tool_invoker=self.tool.invoke,
            tool_limit_getter=self.tool.get_tool_max_calls,
        )

    async def invoke(self, state: GraphState, system_prompt: str | None = None) -> GraphState:
        """프레임워크 진입점"""
        node_name = self.__class__.__name__
        logger.info(f"[{node_name}] 실행 시작")

        self._executor.set_system_prompt(system_prompt)
        start = time.perf_counter()
        try:
            result = await self.run(state)
            elapsed = round(time.perf_counter() - start, 2)
            logger.info(f"[{node_name}] 실행 완료 ({elapsed}s)")
            return result
        except Exception:
            elapsed = round(time.perf_counter() - start, 2)
            logger.exception(f"[{node_name}] 실행 실패 ({elapsed}s)")
            raise

    @abstractmethod
    async def run(self, state: GraphState) -> GraphState:
        """Agent 구현부 — 도메인 로직만 작성"""

    # ── 서비스 접근자 ──

    @property
    def retrieval(self):
        """retrieval 모듈 접근자"""
        from .. import retrieval

        return retrieval

    @property
    def tool(self):
        """tool 모듈 접근자"""
        from .. import tool

        return tool

    # ── 유틸리티 ──

    @staticmethod
    def get_agent_prompt() -> str:
        """Agent identity 프롬프트 반환"""
        from ..manager import get_agent_description

        return get_agent_description()
