"""Tool 예외 정의"""


class ToolNotFoundError(Exception):
    """Tool 카탈로그에 미등록"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' 미존재")


class ToolTimeoutError(Exception):
    """Tool 호출 타임아웃 — MCP Server _meta 기준"""

    def __init__(self, tool_name: str, timeout: float):
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(f"Tool '{tool_name}' 응답 시간 초과 ({timeout}s)")


class ToolConnectionError(Exception):
    """MCP Server 연결 실패 — 재시도 소진 후 발생"""

    def __init__(self, tool_name: str, cause: Exception):
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' 연결 실패: {cause}")
