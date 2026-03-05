"""DefaultResponser — 미분류 요청 처리"""

from ..state import GraphState
from ._base_agent import BaseAgent


class DefaultResponser(BaseAgent):
    """분류 실패 시 LLM 기반 안내 응답 생성"""

    async def run(self, state: GraphState, prompt_template: str = "") -> GraphState:
        prompt = prompt_template.format(query=state.query)
        result = await self.invoke_llm(prompt)
        state.answer = result.strip() if result else "요청을 처리할 수 없습니다."
        return state
