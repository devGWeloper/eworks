"""KnowledgeInquiryAgent — 지식/메뉴얼/기준정보 조회"""

import json

from ...prompt.domain.knowledge_inquiry_prompt import (
    KNOWLEDGE_INQUIRY_SYSTEM_PROMPT,
    KNOWLEDGE_INQUIRY_USER_PROMPT,
)
from ...state import GraphState
from .._base_agent import BaseAgent


class KnowledgeInquiryAgent(BaseAgent):
    """지식, 메뉴얼, 기준정보 조회"""

    system_prompt = KNOWLEDGE_INQUIRY_SYSTEM_PROMPT
    user_prompt_template = KNOWLEDGE_INQUIRY_USER_PROMPT

    intent_id = "KNOWLEDGE_INQUIRY_AGENT"
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

    async def run(self, state: GraphState) -> GraphState:
        from .._domain_registry import get_intent_catalog

        context = state.context
        topic = context.classified_parameters.get("topic", "")
        query = topic or state.query

        # RAG 검색
        knowhow_results = await self.retrieval.knowhow().similarity_search(query=query, k=3)
        upload_results = await self.retrieval.upload().similarity_search(query=query, k=3)

        knowhow_text = "\n".join(doc.page_content for doc in knowhow_results)
        upload_text = "\n".join(doc.page_content for doc in upload_results)

        context.result = await self._executor.execute(
            prompt_vars={
                "agent_prompt": self.get_agent_prompt(),
                "topic": f"{topic}\n\n## Knowhow 검색 결과\n{knowhow_text}\n\n## 문서 검색 결과\n{upload_text}",
                "intent_catalog": json.dumps(get_intent_catalog(), ensure_ascii=False),
            },
        )
        return state
