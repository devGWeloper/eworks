"""에이전트 워크플로우 오케스트레이터"""

import logging
import os

from core.base.decorators import prompt
from core.base.manager import AgentManager, AgentManagerProvider
from core.services.llm_model.llm_model_service_core import LLMModelServiceCore
from langgraph.graph import END

from .state import GraphState

from .config.settings import (  # isort: skip
    DEFAULT_LLM_MODEL,
    LLM_MODELS,
    LLMModelName,
    MCP_SERVERS,
    RETRIEVAL_COLLECTIONS,
    RETRIEVAL_EMBEDDINGS,
)
from .node.classifier import Classifier
from .node.default_responser import DefaultResponser
from .node.domain.analyze_log_node import AnalyzeLogAgent
from .node.domain.knowledge_inquiry_node import KnowledgeInquiryAgent
from .node.normalizer import Normalizer
from .node.responser import Responser
from .node.router import _to_node_name, route_by_intent
from .prompt.default_responser_prompt import DEFAULT_RESPONSER_PROMPT
from .prompt.domain.analyze_log_prompt import ANALYZE_LOG_PROMPT
from .prompt.domain.knowledge_inquiry_prompt import KNOWLEDGE_INQUIRY_PROMPT
from .prompt.normalizer_prompt import NORMALIZER_PROMPT
from .prompt.responser_prompt import RESPONSER_PROMPT
from .retrieval.retrieval_service import RetrievalService
from .tool.tool_service import ToolService

logger = logging.getLogger(__name__)
VERSION = os.path.basename(os.path.dirname(os.path.abspath(__file__))).replace("_", ".")

# ─── 초기화 ───────────────────────────────────────────────────

logger.info("Agent Workflow 초기화 시작")

# 1. LLMModelServiceCore 싱글톤 초기화
LLMModelServiceCore.initialize(models=LLM_MODELS, default_model=DEFAULT_LLM_MODEL)

# 2. RetrievalService 싱글톤 초기화 (Milvus 환경변수 미설정 시 건너뜀)
if os.getenv("MILVUS_URI"):
    RetrievalService.initialize(
        collections=RETRIEVAL_COLLECTIONS,
        embeddings=RETRIEVAL_EMBEDDINGS,
    )
else:
    logger.warning("MILVUS_URI 미설정 — RetrievalService 초기화 건너뜀")

# 3. ToolService 싱글톤 초기화 (async — MCP 서버 연결 + Tool 탐색)
import asyncio

asyncio.run(ToolService.initialize(servers=MCP_SERVERS))

# 4. AgentManager 초기화
AgentManagerProvider.initialize(
    agent_prompt="당신은 반도체 제조 공정의 MES(Manufacturing Execution System) 관련 질의나 처리를 담당하는 Agent입니다.",
    manager=AgentManager(
        state_schema=GraphState,
        service_id="",
        author="",
        workflow_name="",
        description="",
        workflow_version=VERSION,
        main_model=LLMModelName.LLAMA3_70B.value,
    ),
)
manager = AgentManagerProvider.get_manager()

# 5. Node 초기화
normalizer = Normalizer()
classifier = Classifier()
responser = Responser()
knowledge_inquiry_agent = KnowledgeInquiryAgent()
analyze_log_agent = AnalyzeLogAgent()
default_responser = DefaultResponser()


# ─── 노드 함수 ───────────────────────────────────────────────


@prompt(NORMALIZER_PROMPT)
async def run_normalizer(state: GraphState) -> GraphState:
    """Normalizer 실행"""
    logger.info("========== Normalizer 실행 ==========")
    return await normalizer.run(state, run_normalizer.prompt)


async def run_classifier(state: GraphState) -> GraphState:
    """Classifier 실행"""
    logger.info("========== Classifier 실행 ==========")
    return await classifier.run(state)


@prompt(KNOWLEDGE_INQUIRY_PROMPT)
async def run_knowledge_inquiry(state: GraphState) -> GraphState:
    """KnowledgeInquiryAgent 실행"""
    logger.info("========== KnowledgeInquiryAgent 실행 ==========")
    return await knowledge_inquiry_agent.run(state, run_knowledge_inquiry.prompt)


@prompt(ANALYZE_LOG_PROMPT)
async def run_analyze_log(state: GraphState) -> GraphState:
    """AnalyzeLogAgent 실행"""
    logger.info("========== AnalyzeLogAgent 실행 ==========")
    return await analyze_log_agent.run(state, run_analyze_log.prompt)


@prompt(DEFAULT_RESPONSER_PROMPT)
async def run_default_responser(state: GraphState) -> GraphState:
    """DefaultResponser 실행"""
    logger.info("========== DefaultResponser 실행 ==========")
    return await default_responser.run(state, run_default_responser.prompt)


@prompt(RESPONSER_PROMPT)
async def run_responser(state: GraphState) -> GraphState:
    """Responser 실행"""
    logger.info("========== Responser 실행 ==========")
    return await responser.run(state, run_responser.prompt)


# ─── 라우팅 ───────────────────────────────────────────────────

DOMAIN_NODE_FUNCS = {
    _to_node_name(knowledge_inquiry_agent.intent_id): (run_knowledge_inquiry, knowledge_inquiry_agent.intent_name),
    _to_node_name(analyze_log_agent.intent_id): (run_analyze_log, analyze_log_agent.intent_name),
}


# ─── 그래프 구성 및 컴파일 ────────────────────────────────────

manager.add_node(name="normalizer", description="사용자 입력 정규화", func=run_normalizer)
manager.add_node(name="classifier", description="단일 intent 분류", func=run_classifier)

for node_name, (node_func, description) in DOMAIN_NODE_FUNCS.items():
    manager.add_node(name=node_name, description=description, func=node_func)

manager.add_node(name="default_response", description="미분류 처리", func=run_default_responser)
manager.add_node(name="responser", description="최종 응답 생성", func=run_responser)

manager.set_entry_point("normalizer")
manager.add_edge("normalizer", "classifier")

manager.add_conditional_edges(
    "classifier",
    route_by_intent,
    {node_name: node_name for node_name in list(DOMAIN_NODE_FUNCS.keys()) + ["default_response"]},
)

for node_name in DOMAIN_NODE_FUNCS:
    manager.add_edge(node_name, "responser")
manager.add_edge("default_response", "responser")
manager.add_edge("responser", END)

graph = manager.compile()
logger.info("Agent Workflow 컴파일 완료")
