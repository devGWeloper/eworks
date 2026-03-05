"""Tool 모듈 — MCP Server 연결 관리 + Tool 카탈로그"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config.settings import MCPServerConfig
from .exceptions import ToolConnectionError, ToolNotFoundError, ToolTimeoutError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2]
DEFAULT_MAX_CALLS = 10
DEFAULT_TIMEOUT = 60.0

# ── 모듈 레벨 상태 ──

_server_configs: Dict[str, MCPServerConfig] = {}
_tool_catalog: Dict[str, "ToolInfo"] = {}
_connected: bool = False


@dataclass
class ToolInfo:
    """Tool 카탈로그 엔트리 — MCP Server _meta에서 제한값 자동 수집"""

    server_key: str
    schema: Dict[str, Any] = field(default_factory=dict)
    max_calls: int = DEFAULT_MAX_CALLS
    timeout: float = DEFAULT_TIMEOUT


# ── 초기화 ──


def initialize(servers: Dict[str, MCPServerConfig]) -> None:
    """앱 시작 시 1회 호출 — config 저장 + warmup"""
    global _server_configs, _tool_catalog, _connected

    if not servers:
        raise ValueError("servers는 최소 하나 이상의 MCP Server 설정을 포함해야 합니다")

    _server_configs = dict(servers)
    _tool_catalog = {}
    _connected = False

    logger.info(f"Tool 초기화 완료 (서버 설정: {list(servers.keys())})")

    # warmup — 이벤트 루프 존재 시 background task로 MCP 연결 시도
    try:
        asyncio.get_running_loop().create_task(warmup())
    except RuntimeError:
        pass  # 이벤트 루프 미존재 시 첫 사용 시점에 연결


async def warmup() -> None:
    """MCP Server 연결 시도 — 실패해도 앱 기동 차단 안함"""
    try:
        await _ensure_tools()
    except Exception as e:
        logger.warning(f"Tool warmup 실패: {e}. 첫 Tool 사용 시 재연결 시도")


# ── 연결 관리 ──


async def _ensure_tools() -> None:
    """미연결 시 MCP Server 연결 + Tool 탐색"""
    global _connected

    if _connected:
        return

    failed_servers = []
    for server_key, config in _server_configs.items():
        try:
            await _discover_tools(server_key, config)
        except Exception as e:
            logger.warning(f"MCP Server '{server_key}' 연결 실패 — Tool 비활성화 (error: {e})")
            failed_servers.append(server_key)

    _connected = True

    if failed_servers:
        logger.warning(f"연결 실패 서버: {failed_servers}")

    logger.info(
        f"Tool 연결 완료 "
        f"(서버: {len(_server_configs) - len(failed_servers)}/{len(_server_configs)} 성공, "
        f"도구: {list(_tool_catalog.keys())})"
    )


async def _reconnect() -> None:
    """MCP Server 재연결 — 카탈로그 초기화 후 재탐색"""
    global _tool_catalog, _connected

    logger.warning("Tool 재연결 시도")
    _tool_catalog = {}
    _connected = False
    await _ensure_tools()


async def _execute_with_retry(operation: Callable) -> Any:
    """재시도 래퍼 — 실패 시 재연결 + 백오프, ToolNotFoundError는 즉시 전파"""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if attempt == 0:
                await _ensure_tools()
            else:
                wait = BACKOFF_SECONDS[attempt - 1]
                logger.warning(f"Tool 재시도 {attempt + 1}/{MAX_RETRIES} ({wait}s 대기)")
                await asyncio.sleep(wait)
                await _reconnect()

            return await operation()

        except ToolNotFoundError:
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"Tool 작업 실패 (시도 {attempt + 1}/{MAX_RETRIES}): {e}")

    raise last_error


# ── 카탈로그 ──


def list_tools() -> List[str]:
    """등록된 전체 Tool 이름 목록"""
    return list(_tool_catalog.keys())


def get_tool_schema(tool_name: str) -> Optional[Dict]:
    """Tool의 JSON Schema 반환 (없으면 None)"""
    info = _tool_catalog.get(tool_name)
    if info is None:
        return None
    return info.schema


def has_tool(tool_name: str) -> bool:
    """Tool 존재 여부"""
    return tool_name in _tool_catalog


def get_tool_max_calls(tool_name: str) -> int:
    """Tool별 최대 호출 횟수 (미등록 시 기본값)"""
    info = _tool_catalog.get(tool_name)
    return info.max_calls if info else DEFAULT_MAX_CALLS


def get_tool_timeout(tool_name: str) -> float:
    """Tool별 호출 타임아웃 초 (미등록 시 기본값)"""
    info = _tool_catalog.get(tool_name)
    return info.timeout if info else DEFAULT_TIMEOUT


async def get_tools_for_binding(tool_names: List[str]) -> List[Dict]:
    """LLM bind_tools()용 tool definition 목록 반환, 미존재 tool 제외"""
    await _ensure_tools()

    result = []
    for name in tool_names:
        schema = get_tool_schema(name)
        if schema is None:
            logger.warning(f"Tool '{name}' 미존재 — bind_tools 목록에서 제외")
            continue
        result.append(schema)
    return result


# ── 호출 ──


async def invoke(tool_name: str, params: Dict) -> Dict:
    """Tool 호출 — 재시도 래퍼 적용, reconnect 후 최신 카탈로그 참조

    Raises:
        ToolNotFoundError: 카탈로그 미등록 (재시도 없이 즉시)
        ToolTimeoutError: 호출 타임아웃 (재시도 소진 후)
        ToolConnectionError: MCP Server 연결 실패 (재시도 소진 후)
    """

    async def _do_invoke():
        info = _tool_catalog.get(tool_name)
        if info is None:
            raise ToolNotFoundError(tool_name)

        logger.info(f"Tool '{tool_name}' 호출 (server: {info.server_key}, params: {params})")
        try:
            result = await asyncio.wait_for(info.schema.ainvoke(params), timeout=info.timeout)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(tool_name, info.timeout) from None
        logger.info(f"Tool '{tool_name}' 결과: {result}")
        return result

    try:
        return await _execute_with_retry(_do_invoke)
    except (ToolNotFoundError, ToolTimeoutError):
        raise
    except Exception as e:
        raise ToolConnectionError(tool_name, e) from e


# ── 내부 ──


async def _discover_tools(server_key: str, config: MCPServerConfig) -> None:
    """MCP Server 단일 연결 시도 + Tool 카탈로그 등록"""
    logger.info(f"MCP Server '{server_key}' 연결 시도 (url: {config.url}, transport: {config.transport})")
    mcp_client = MultiServerMCPClient({server_key: {"transport": config.transport, "url": config.url}})
    tools = await mcp_client.get_tools()

    if tools is not None:
        for t in tools:
            if t.name in _tool_catalog:
                existing = _tool_catalog[t.name]
                logger.warning(
                    f"Tool '{t.name}' 중복 등록 — 기존: {existing.server_key}, 신규: {server_key} (덮어쓰기)"
                )

            meta = (getattr(t, "metadata", None) or {}).get("_meta", {})
            max_calls = meta.get("max_calls", DEFAULT_MAX_CALLS)
            timeout = meta.get("timeout", DEFAULT_TIMEOUT)

            _tool_catalog[t.name] = ToolInfo(server_key, t, max_calls, timeout)
            logger.debug(f"Tool '{t.name}' 등록 (max_calls={max_calls}, timeout={timeout}s)")

    logger.info(f"MCP Server '{server_key}' 연결 성공")


def available_servers() -> List[str]:
    """등록된 MCP Server 이름 목록"""
    return list(_server_configs.keys())
