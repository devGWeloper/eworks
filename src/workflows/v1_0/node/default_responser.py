"""DefaultResponser — 미분류 요청 처리"""

from ..prompt.default_responser_prompt import (
    DEFAULT_RESPONSER_SYSTEM_PROMPT,
    DEFAULT_RESPONSER_USER_PROMPT,
)
from ..state import GraphState
from ._base_agent import BaseAgent


class DefaultResponser(BaseAgent):
    """분류 실패 시 LLM 기반 안내 응답 생성"""

    system_prompt = DEFAULT_RESPONSER_SYSTEM_PROMPT
    user_prompt_template = DEFAULT_RESPONSER_USER_PROMPT

    async def run(self, state: GraphState) -> GraphState:
        result = await self._executor.execute(
            prompt_vars={"query": state.query},
        )
        state.answer = result.strip() if result else "요청을 처리할 수 없습니다."
        return state
