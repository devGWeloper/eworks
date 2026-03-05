"""도메인 Agent 레지스트리 — 단일 등록 지점"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_domain_agent_classes() -> tuple:
    """등록된 도메인 Agent 클래스 반환 (lazy import)"""
    from .domain.analyze_log_node import AnalyzeLogAgent
    from .domain.knowledge_inquiry_node import KnowledgeInquiryAgent

    return (KnowledgeInquiryAgent, AnalyzeLogAgent)


@lru_cache(maxsize=1)
def get_intent_catalog() -> dict:
    """Classifier 프롬프트용 intent 카탈로그 파생"""
    return {
        cls.intent_id: {
            "intent_name": cls.intent_name,
            "intent_description": cls.intent_description,
            "intent_examples": cls.intent_examples,
            "intent_parameters": cls.intent_parameters,
        }
        for cls in _get_domain_agent_classes()
    }
