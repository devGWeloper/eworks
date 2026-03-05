"""AnalyzeLogAgent — Tool Calling 기반 존재 여부 확인"""

from ...prompt.domain.analyze_log_prompt import (
    ANALYZE_LOG_SYSTEM_PROMPT,
    ANALYZE_LOG_USER_PROMPT,
)
from ...state import GraphState
from .._base_agent import BaseAgent


class AnalyzeLogAgent(BaseAgent):
    """특정 대상의 존재 여부를 모니터링 데이터에서 확인"""

    system_prompt = ANALYZE_LOG_SYSTEM_PROMPT
    user_prompt_template = ANALYZE_LOG_USER_PROMPT

    intent_id = "ANALYZE_LOG_AGENT"
    intent_name = "존재 여부 확인"
    intent_description = "특정 대상의 존재 여부를 모니터링 데이터에서 확인"
    intent_examples = [
        "장비 A에서 START 이벤트 발생했는지 확인해줘",
        "센서 Core에서 ERROR 알람 있었어?",
        "오늘 PM1 가동 기록 있어?",
    ]
    intent_parameters = [
        {"name": "target", "type": "str", "required": True, "description": "확인 대상"},
        {"name": "condition", "type": "str", "required": False, "description": "확인 조건"},
    ]

    def get_allowed_tools(self):
        return ["monitoring_check_tool", "monitoring_history_tool"]

    async def run(self, state: GraphState) -> GraphState:
        context = state.context
        target = context.classified_parameters.get("target", "unknown")
        condition = context.classified_parameters.get("condition", "")

        tools = await self.tool.get_tools_for_binding(self.get_allowed_tools())

        result = await self._executor.execute_with_tools(
            tools=tools,
            prompt_vars={
                "agent_prompt": self.get_agent_prompt(),
                "target": target,
                "condition": condition,
            },
            tool_limits={
                "monitoring_check_tool": {"max_calls": 5},
            },
        )
        context.result = result["content"]
        return state
