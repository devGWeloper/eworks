"""Normalizer — 사용자 입력 정규화 Agent"""

import re

from ..prompt.normalizer_prompt import NORMALIZER_SYSTEM_PROMPT, NORMALIZER_USER_PROMPT
from ..state import GraphState
from ._base_agent import BaseAgent


class Normalizer(BaseAgent):
    """사용자 입력 정규화 Agent"""

    system_prompt = NORMALIZER_SYSTEM_PROMPT
    user_prompt_template = NORMALIZER_USER_PROMPT

    async def run(self, state: GraphState) -> GraphState:
        # 텍스트 정규화 (공백/개행/제어문자 정리)
        text = state.query.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        # 대화 이력 기반 축약 + 대명사 해소 (LLM)
        result = await self._executor.execute(
            prompt_vars={
                "chat_history": state.chat_history or [],
                "query": state.query,
                "normalized": text,
            },
        )
        state.context.normalized_input = result.strip() if result else text
        return state
