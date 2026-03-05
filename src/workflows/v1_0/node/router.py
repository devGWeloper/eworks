"""조건부 라우팅 — intent_id 기반 도메인 노드 분기"""

from ..state import GraphState
from ._domain_registry import get_intent_catalog


def _to_node_name(intent_id: str) -> str:
    """intent_id → LangGraph 노드명 변환 (Mermaid 호환)"""
    return intent_id.replace(".", "_")


def route_by_intent(state: GraphState) -> str:
    """분류된 intent_id 기반 도메인 노드 라우팅"""
    intent_id = state.context.classified_intent_id
    if intent_id in get_intent_catalog():
        return _to_node_name(intent_id)
    return "default_response"
