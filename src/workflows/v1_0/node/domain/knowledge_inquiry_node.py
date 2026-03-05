"""KnowledgeInquiryAgent — 지식/메뉴얼/기준정보 조회 [Preset]"""

import json

from ...state import GraphState
from .._base_agent import BaseAgent


class KnowledgeInquiryAgent(BaseAgent):
    """지식, 메뉴얼, 기준정보 조회"""

    intent_id = "KNOWLEDGE.INQUIRY"
    intent_name = "지식 조회"
    intent_description = "지식, 메뉴얼, 기준정보 또는 Agent 기능에 대한 조회"
    intent_examples = [
        "뭘 할 수 있어?",
        "기능 목록 알려줘",
        "CVD 공정 설명해줘",
        "이 장비 메뉴얼 보여줘",
    ]
    intent_parameters = [
        {"name": "topic", "type": "str", "required": True, "description": "조회 대상"},
    ]

    async def run(self, state: GraphState, prompt_template: str = "") -> GraphState:
        from .._domain_registry import get_intent_catalog

        context = state.context
        topic = context.classified_parameters.get("topic", "")

        prompt = prompt_template.format(
            agent_prompt=self.get_agent_prompt(),
            topic=topic,
            intent_catalog=json.dumps(get_intent_catalog(), ensure_ascii=False),
        )

        context.result = await self.invoke_llm(prompt)
        return state
