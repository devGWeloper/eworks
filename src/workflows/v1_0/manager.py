"""AgentManager 확장 + 전역 접근"""

from .temp.core import AgentManager as _BaseAgentManager


class AgentManager(_BaseAgentManager):
    """AgentManager 확장 — agent_description 추가"""

    def __init__(self, *, agent_description: str = "", **kwargs):
        super().__init__(**kwargs)
        self.agent_description = agent_description


_manager: AgentManager | None = None


def initialize(manager: AgentManager) -> AgentManager:
    global _manager
    _manager = manager
    return _manager


def get_manager() -> AgentManager:
    if _manager is None:
        raise RuntimeError("AgentManager not initialized.")
    return _manager


def get_agent_description() -> str:
    return get_manager().agent_description


def middle_stream_text(text: str, name: str = "", description: str = "") -> bool:
    """node에서 로깅처럼 사용하는 편의 함수"""
    return get_manager().middle_stream_text(text, name, description)
