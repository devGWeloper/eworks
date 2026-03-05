"""ToolService — Tool Service [Preset]"""

from core.services.tool.tool_service_core import ToolServiceCore


class ToolService(ToolServiceCore):
    """Tool Service [Preset] 확장점

    Agent 개발자가 도메인 특화 메서드를 추가하는 확장점
    현재는 ToolServiceCore 기능을 그대로 노출
    """

    pass
