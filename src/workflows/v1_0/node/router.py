"""조건부 라우팅 — intent_id 기반 도메인 노드 분기"""

from ..state import GraphState
from ._domain_registry import get_intent_catalog


def route_by_intent(state: GraphState) -> str:
    """분류된 intent_id 기반 도메인 노드 라우팅"""
    intent_id = state.context.classified_intent_id
    if intent_id in get_intent_catalog():
        return intent_id
    return "default_response"
