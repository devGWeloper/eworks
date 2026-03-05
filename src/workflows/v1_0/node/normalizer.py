"""Normalizer — 사용자 입력 정규화 Agent

역할:
  1. 원시 텍스트 정규화 (공백/개행/제어문자 정리)
  2. 대화 이력 기반 대명사 해소 및 질의 재작성 (LLM)
  3. 긴 입력 축약
"""

import re

from ..state import GraphState
from ._base_agent import BaseAgent


class Normalizer(BaseAgent):
    """사용자 입력 정규화 Agent"""

    async def run(self, state: GraphState, prompt_template: str = "") -> GraphState:
        # 텍스트 정규화 (공백/개행/제어문자 정리)
        text = state.query.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        # 대화 이력 기반 축약 + 대명사 해소 (LLM)
        prompt = prompt_template.format(
            chat_history=state.chat_history or [],
            query=state.query,
            normalized=text,
        )
        result = await self.invoke_llm(prompt)
        state.context.normalized_input = result.strip() if result else text
        return state
