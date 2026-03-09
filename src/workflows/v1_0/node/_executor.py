"""AgentExecutor — LLM 실행기 (텍스트 생성 + Tool Calling)"""

import json
import logging
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..core.llm import call_llm

logger = logging.getLogger(__name__)


class _SafeDict(dict):
    """format_map용 — 없는 키는 {key} 원형 유지 (safe_substitute 동작과 동일)"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class AgentExecutor:
    """프롬프트 조립 → LLM 호출 → 응답 반환

    Args:
        system_prompt: 시스템 프롬프트 (Template 변수 지원)
        user_prompt_template: 사용자 프롬프트 템플릿 (Template 변수 지원)
        model_name_key: LLM 모델 이름, None이면 기본 모델 사용
        tool_invoker: Tool 호출 콜백 — async (tool_name, params) -> result
        tool_limit_getter: Tool별 최대 호출 횟수 조회 콜백 — (tool_name) -> int
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        model_name_key: Optional[str] = None,
        tool_invoker: Optional[Callable] = None,
        tool_limit_getter: Optional[Callable] = None,
    ):
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._model_name_key = model_name_key
        self._tool_invoker = tool_invoker
        self._tool_limit_getter = tool_limit_getter
        self._active_system_prompt: Optional[str] = None

    # ── system_prompt 관리 ──

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        """invoke() 호출 시 자동 설정 — 클래스 기본 system_prompt 대체"""
        self._active_system_prompt = system_prompt

    # ── 메시지 조립 ──

    def _build_messages(self, prompt_vars: Optional[Dict] = None) -> list:
        """system_prompt + user_prompt_template → 메시지 리스트 조립"""
        extra = prompt_vars or {}
        system_prompt = self._active_system_prompt or self._system_prompt
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt.format_map(_SafeDict(extra))))

        if self._user_prompt_template:
            messages.append(HumanMessage(content=self._user_prompt_template.format_map(_SafeDict(extra))))

        return messages

    # ── 텍스트 생성 ──

    async def execute(self, prompt_vars: Optional[Dict] = None) -> str:
        """프롬프트 조립 → LLM 호출 → 텍스트 응답 반환

        Args:
            prompt_vars: 프롬프트 Template 치환 변수
        """
        messages = self._build_messages(prompt_vars)
        response = await call_llm(messages, model_name_key=self._model_name_key)
        return response.content

    # ── Tool Calling ──

    async def execute_with_tools(
        self,
        tools: List[Dict],
        prompt_vars: Optional[Dict] = None,
        max_total_calls: int = 30,
        tool_limits: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """Tool Calling 루프 — 프롬프트 조립 + LLM 호출 + Tool 실행 반복

        per-tool 제한 우선순위:
        tool_limits[name]["max_calls"] > tool_limit_getter(name) > 기본값 10

        Args:
            tools: bind_tools용 tool schema 리스트
            prompt_vars: 프롬프트 Template 치환 변수
            max_total_calls: 전체 tool 최대 호출 횟수
            tool_limits: Agent 오버라이드 — {"tool_name": {"max_calls": N}}
        """
        if self._tool_invoker is None:
            raise RuntimeError("tool_invoker 미설정. AgentExecutor 생성 시 tool_invoker를 전달하세요")

        messages = self._build_messages(prompt_vars)
        overrides = tool_limits or {}
        tool_call_counts: Dict[str, int] = {}
        total_calls = 0

        while True:
            response = await call_llm(messages, tools=tools, model_name_key=self._model_name_key)

            if not response.tool_calls:
                return {
                    "status": "success",
                    "content": response.content,
                    "tool_call_count": total_calls,
                }

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                total_calls += 1

                # per-tool 제한: 오버라이드 > 서버 기본값 > 하드코딩 기본값
                per_tool_limit = overrides.get(tool_name, {}).get("max_calls")
                if per_tool_limit is None:
                    per_tool_limit = self._tool_limit_getter(tool_name) if self._tool_limit_getter else 10

                if tool_call_counts[tool_name] > per_tool_limit:
                    logger.warning(f"Tool '{tool_name}' 호출 횟수 초과 ({per_tool_limit}회)")
                    return {
                        "status": "failed",
                        "content": f"Tool '{tool_name}' 호출 횟수 초과 ({per_tool_limit}회)",
                        "tool_call_count": total_calls,
                    }

                if total_calls > max_total_calls:
                    logger.warning(f"전체 Tool 호출 횟수 초과 ({max_total_calls}회)")
                    return {
                        "status": "failed",
                        "content": f"전체 Tool 호출 횟수 초과 ({max_total_calls}회)",
                        "tool_call_count": total_calls,
                    }

                try:
                    tool_result = await self._tool_invoker(tool_name, tool_args)
                except Exception as e:
                    logger.warning(f"Tool '{tool_name}' 호출 실패: {e}")
                    tool_result = {"error": str(e)}

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        tool_call_id=tool_call["id"],
                    )
                )
