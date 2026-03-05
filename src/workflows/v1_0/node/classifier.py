"""Classifier — 사용자 입력 의도 분류"""

import json
import logging
import re
from typing import Any, Dict, Tuple

from ..prompt.classifier_prompt import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_PROMPT
from ..state import GraphState
from ._base_agent import BaseAgent
from ._domain_registry import get_intent_catalog

logger = logging.getLogger(__name__)


class Classifier(BaseAgent):
    """사용자 입력 의도 분류

    normalized_input → LLM 분류 → classified_intent_id + classified_parameters
    """

    system_prompt = CLASSIFIER_SYSTEM_PROMPT
    user_prompt_template = CLASSIFIER_USER_PROMPT

    async def run(self, state: GraphState) -> GraphState:
        context = state.context

        # LLM 호출 (AgentExecutor 경유)
        intent_catalog = json.dumps(get_intent_catalog(), ensure_ascii=False, indent=2)
        result = await self._executor.execute(
            prompt_vars={
                "user_input": context.normalized_input,
                "intent_catalog": intent_catalog,
            },
        )

        # 응답 파싱
        intent_id, parameters = self._parse_response(result)
        context.classified_intent_id = intent_id
        context.classified_parameters = parameters

        logger.info(f"분류 결과: intent_id={intent_id}, params={parameters}")
        return state

    # ── 응답 파싱 ──

    def _parse_response(self, llm_response: str) -> Tuple[str, Dict[str, Any]]:
        """LLM 응답에서 intent_id, parameters 추출"""
        try:
            json_text = self._extract_json(llm_response)
            data = json.loads(json_text)
            return (data.get("intent_id", "unknown"), data.get("parameters", {}))

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}, response={llm_response[:200]}")
            return ("unknown", {})

    @staticmethod
    def _extract_json(text: str) -> str:
        """LLM 응답에서 JSON 문자열 추출 (markdown code block 처리)"""
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text
