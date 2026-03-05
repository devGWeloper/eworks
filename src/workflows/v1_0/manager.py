"""AgentManagerProvider — 워크플로우 매니저 싱글톤 제공"""

from .temp.gaia import AgentManager


class AgentManagerProvider:
    _manager: AgentManager | None = None
    _agent_prompt: str = ""

    @classmethod
    def initialize(cls, manager: AgentManager, agent_prompt: str = "") -> None:
        cls._manager = manager
        cls._agent_prompt = agent_prompt

    @classmethod
    def middle_stream_text(cls, text: str, name: str = "", description: str = "") -> bool:
        if cls._manager is None:
            raise RuntimeError("AgentManager not initialized.")
        return cls._manager.middle_stream_text(text, name, description)

    @classmethod
    def get_manager(cls) -> AgentManager:
        if cls._manager is None:
            raise RuntimeError("AgentManager not initialized.")
        return cls._manager

    @classmethod
    def get_agent_prompt(cls) -> str:
        """Agent identity 프롬프트 반환"""
        return cls._agent_prompt
