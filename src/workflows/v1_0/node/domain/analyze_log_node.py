"""AnalyzeLogAgent — 존재 여부 확인"""

import json

from ...state import GraphState
from .._base_agent import BaseAgent


class AnalyzeLogAgent(BaseAgent):
    """특정 대상의 존재 여부를 모니터링 데이터에서 확인"""

    intent_id = "ANALYZE.LOG"
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
        return ["monitoring_check_tool"]

    async def run(self, state: GraphState, prompt_template: str = "") -> GraphState:
        context = state.context
        target = context.classified_parameters.get("target", "unknown")
        condition = context.classified_parameters.get("condition", "")

        # TODO: 실제 존재 여부 확인 로직 구현
        output = json.dumps(
            {
                "exists": True,
                "target": target,
                "condition": condition,
                "found_count": 0,
                "message": f"'{target}'에 대한 존재 여부 확인 완료",
            },
            ensure_ascii=False,
        )

        context.result = output
        return state
