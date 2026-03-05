"""v1_2 워크플로우 상태 정의"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class AgentContext:
    """워크플로우 내부 처리용 컨텍스트"""

    normalized_input: Optional[str] = None
    classified_intent_id: Optional[str] = None
    classified_parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None


class GGraphState(BaseModel):
    query: str
    chat_history: list[str] | None
    answer: str | None
    session_id: str


class GraphState(GGraphState):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: AgentContext = Field(
        description="Agent Workflow를 통해 흐르는 주요 Context 객체",
        default_factory=lambda: AgentContext(),
    )
