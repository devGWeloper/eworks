"""FastAPI 기반 AI Agent Core 애플리케이션"""

import importlib
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# src/ 디렉토리를 sys.path에 추가하여 기존 import 경로 유지
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from core.base.models import AgentContext
from core.base.state import GraphState

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 버전별 컴파일된 그래프를 저장하는 딕셔너리
workflow_graphs: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """서버 시작 시 워크플로우를 로드합니다."""
    logger.info("AI Agent Core를 시작합니다...")
    discover_and_load_workflows()
    logger.info(f"등록된 워크플로우 버전: {list(workflow_graphs.keys())}")
    yield


app = FastAPI(title="AI Agent Core", lifespan=lifespan)


def parse_version(folder_name: str) -> str | None:
    """
    워크플로우 폴더명을 버전 문자열로 변환합니다.
    예: 'v1_1' -> '1.1.0'
    """
    match = re.match(r"^v(\d+)_(\d+)$", folder_name)
    if not match:
        return None
    major, minor = match.groups()
    return f"{major}.{minor}.0"


def discover_and_load_workflows():
    """
    src/workflow/ 하위 폴더를 스캔하여 워크플로우 버전을 등록하고
    각 버전의 workflow.py를 동적 import하여 그래프를 컴파일합니다.
    """
    workflow_dir = SRC_DIR / "workflows"

    for entry in sorted(workflow_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        version = parse_version(entry.name)
        if version is None:
            continue

        workflow_module_path = entry / "workflow.py"
        if not workflow_module_path.exists():
            logger.warning(f"workflow.py not found in {entry.name}, skipping")
            continue

        module_name = f"workflows.{entry.name}.workflow"
        try:
            module = importlib.import_module(module_name)
            graph = getattr(module, "graph", None)
            if graph is None:
                logger.warning(f"{module_name}에서 graph를 찾을 수 없습니다, skipping")
                continue

            workflow_graphs[version] = graph
            logger.info(f"Workflow v{version} 등록 완료 (from {entry.name})")
        except Exception as e:
            logger.exception(f"Workflow {entry.name} 로드 실패: {e}")
            raise


class CompletionRequest(BaseModel):
    message: str
    user_id: str


@app.post("/api/{version}/api/completion")
async def completion(version: str, request: CompletionRequest):
    """워크플로우 버전에 맞는 LangGraph를 실행합니다."""
    graph = workflow_graphs.get(version)
    if graph is None:
        available = list(workflow_graphs.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version}' not found. Available: {available}",
        )

    context = AgentContext()
    initial_state: GraphState = {
        "query": request.message,
        "chat_history": [],
        "session_id": request.user_id,
        "context": context,
        "answer": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
        return {
            "answer": final_state.get("answer", ""),
            "session_id": request.user_id,
        }
    except Exception as e:
        logger.exception(f"워크플로우 실행 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
