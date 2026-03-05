"""BaseAgent — v1_2 Agent 추상 기본 클래스"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..state import GraphState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent 추상 기본 클래스 — run(state) 단일 인터페이스 + LLM 유틸리티"""

    intent_id: str = ""
    intent_name: str = ""
    intent_description: str = ""
    intent_examples: list = []
    intent_parameters: list = []

    @abstractmethod
    async def run(self, state: GraphState) -> GraphState:
        """state를 받아 처리한 뒤 갱신된 state를 반환"""

    # ── LLM 유틸리티 ──

    def get_llm_model(self, model_name_key: Optional[str] = None):
        """LLM 모델 반환"""
        from core.services.llm_model.llm_model_service_core import LLMModelServiceCore

        factory = LLMModelServiceCore.instance()
        return factory.get_model(model_name_key) if model_name_key else factory.get_default_model()

    @staticmethod
    def get_agent_prompt() -> str:
        """Agent identity 프롬프트 반환"""
        from core.base.manager import AgentManagerProvider

        return AgentManagerProvider.get_agent_prompt()

    async def invoke_llm(self, prompt: str, model_name_key: Optional[str] = None) -> str:
        """LLM 단순 호출"""
        llm = self.get_llm_model(model_name_key)
        response = await llm.ainvoke(prompt)
        return response.content

    async def invoke_llm_with_tools(
        self,
        messages: list,
        tools: List[Dict],
        max_calls_per_tool: int = 10,
        max_total_calls: int = 30,
        model_name_key: Optional[str] = None,
    ) -> Dict:
        """LLM Tool Calling 루프"""
        from core.services.tool.tool_service_core import ToolServiceCore
        from langchain_core.messages import ToolMessage, message_to_dict

        llm = self.get_llm_model(model_name_key)
        llm_with_tools = llm.bind_tools(tools)

        tool_call_counts: Dict[str, int] = {}
        total_calls = 0

        while True:
            response = await llm_with_tools.ainvoke(messages)

            if not response.tool_calls:
                return {
                    "status": "success",
                    "content": response.content,
                    "tool_call_count": total_calls,
                }

            messages.append(message_to_dict(response)["data"])

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                total_calls += 1

                if tool_call_counts[tool_name] > max_calls_per_tool:
                    logger.warning(f"Tool '{tool_name}' 호출 횟수 초과 ({max_calls_per_tool}회)")
                    return {
                        "status": "failed",
                        "content": f"Tool '{tool_name}' 호출 횟수 초과 ({max_calls_per_tool}회)",
                        "tool_call_count": total_calls,
                    }

                if total_calls > max_total_calls:
                    logger.warning(f"전체 Tool 호출 횟수 초과 ({max_total_calls}회)")
                    return {
                        "status": "failed",
                        "content": f"전체 Tool 호출 횟수 초과 ({max_total_calls}회)",
                        "tool_call_count": total_calls,
                    }

                tool_result = await ToolServiceCore.instance().invoke(tool_name, tool_args)

                messages.append(
                    message_to_dict(
                        ToolMessage(
                            content=json.dumps(tool_result, ensure_ascii=False),
                            tool_call_id=tool_call["id"],
                        )
                    )["data"]
                )
