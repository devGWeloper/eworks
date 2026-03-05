"""Responser — 최종 응답 생성 Agent

역할:
  1. 핸들러 실행 결과 종합
  2. 사용자 질의에 대한 최종 답변 생성 (LLM)
"""

from ..state import GraphState
from ._base_agent import BaseAgent


class Responser(BaseAgent):
    """최종 응답 생성 Agent"""

    async def run(self, state: GraphState, prompt_template: str = "") -> GraphState:
        context = state.context
        results_summary = context.result if context.result else "처리된 결과가 없습니다."

        # 최종 응답 생성 (LLM)
        prompt = prompt_template.format(
            results_summary=results_summary,
            query=state.query,
        )
        result = await self.invoke_llm(prompt)
        state.answer = result.strip() if result else "요청을 처리했으나 응답을 생성하지 못했습니다."
        return state
