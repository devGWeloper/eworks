"""Responser — 최종 응답 생성 Agent"""

import logging

from ..prompt.responser_prompt import RESPONSER_SYSTEM_PROMPT, RESPONSER_USER_PROMPT
from ..state import GraphState
from ._base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Responser(BaseAgent):
    """최종 응답 생성 Agent"""

    system_prompt = RESPONSER_SYSTEM_PROMPT
    user_prompt_template = RESPONSER_USER_PROMPT

    async def run(self, state: GraphState) -> GraphState:
        context = state.context
        results_summary = context.result if context.result else "처리된 결과가 없습니다."

        # 최종 응답 생성 (LLM)
        result = await self._executor.execute(
            prompt_vars={
                "results_summary": results_summary,
                "query": state.query,
            },
        )
        state.answer = result.strip() if result else "요청을 처리했으나 응답을 생성하지 못했습니다."
        logger.info(f"최종 응답:\n{state.answer}")
        return state
