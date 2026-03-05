"""에이전트 워크플로우 오케스트레이터"""

import logging
import os

from . import manager as agent_manager
from .core.llm import initialize as init_llm
from .core.logging import setup_logging
from .manager import AgentManager
from .state import GraphState
from .temp.core.constants import GAIA_STANDARD_OUTPUT_NODE

from .config.settings import (  # isort: skip
    DEFAULT_LLM_MODEL,
    LLM_MODELS,
    LLMModel,
    MCP_SERVERS,
    RETRIEVAL_COLLECTIONS,
)

from . import retrieval, tool
from .node.classifier import Classifier
from .node.default_responser import DefaultResponser
from .node.domain.analyze_log_node import AnalyzeLogAgent
from .node.domain.knowledge_inquiry_node import KnowledgeInquiryAgent
from .node.normalizer import Normalizer
from .node.responser import Responser
from .node.router import route_by_intent
from .prompt.default_responser_prompt import DEFAULT_RESPONSER_SYSTEM_PROMPT
from .prompt.domain.analyze_log_prompt import ANALYZE_LOG_SYSTEM_PROMPT
from .prompt.domain.knowledge_inquiry_prompt import KNOWLEDGE_INQUIRY_SYSTEM_PROMPT
from .prompt.normalizer_prompt import NORMALIZER_SYSTEM_PROMPT
from .prompt.responser_prompt import RESPONSER_SYSTEM_PROMPT
from .temp import config

logger = logging.getLogger(__name__)
VERSION = os.path.basename(os.path.dirname(os.path.abspath(__file__))).replace("_", ".")

# ─── 초기화 ───────────────────────────────────────────────────

setup_logging()
logger.info("Agent Workflow 초기화 시작")

# 1. LLM 초기화
init_llm(models=LLM_MODELS, default_model=DEFAULT_LLM_MODEL)

# 2. Retrieval 초기화
retrieval.initialize(
    collections=RETRIEVAL_COLLECTIONS,
)

# 3. Tool 초기화
tool.initialize(servers=MCP_SERVERS)

# 4. AgentManager 초기화
manager = agent_manager.initialize(
    AgentManager(
        agent_description="당신은 반도체 제조 공정의 MES(Manufacturing Execution System) 관련 질의나 처리를 담당하는 Agent입니다.",
        state_schema=GraphState,
        service_id=config.SERVICE_ID,
        author="FAB MOS",
        workflow_name="MES 운영 지능화 Agent",
        description="Prompt 주입 및 Intent 정적 Router 적용",
        workflow_version=VERSION,
        main_model=LLMModel.QWEN3.value,
    ),
)

# 5. Node 초기화
normalizer = Normalizer()
classifier = Classifier()
default_responser = DefaultResponser()
responser = Responser()

analyze_log_agent = AnalyzeLogAgent()
knowledge_inquiry_agent = KnowledgeInquiryAgent()


# ─── 노드 함수 ───────────────────────────────────────────────


@manager.main_model_type()
@manager.prompt_type(prompt=NORMALIZER_SYSTEM_PROMPT)
async def run_normalizer(state: GraphState) -> GraphState:
    """Normalizer 실행"""
    return await normalizer.invoke(state, system_prompt=run_normalizer.prompt)


async def run_classifier(state: GraphState) -> GraphState:
    """Classifier 실행"""
    return await classifier.invoke(state)


@manager.prompt_type(prompt=DEFAULT_RESPONSER_SYSTEM_PROMPT)
async def run_default_responser(state: GraphState) -> GraphState:
    """DefaultResponser 실행"""
    return await default_responser.invoke(state, system_prompt=run_default_responser.prompt)


@manager.prompt_type(prompt=RESPONSER_SYSTEM_PROMPT)
async def run_responser(state: GraphState) -> GraphState:
    """Responser 실행"""
    return await responser.invoke(state, system_prompt=run_responser.prompt)


@manager.prompt_type(prompt=ANALYZE_LOG_SYSTEM_PROMPT)
async def run_analyze_log(state: GraphState) -> GraphState:
    """AnalyzeLogAgent 실행"""
    return await analyze_log_agent.invoke(state, system_prompt=run_analyze_log.prompt)


@manager.prompt_type(prompt=KNOWLEDGE_INQUIRY_SYSTEM_PROMPT)
async def run_knowledge_inquiry(state: GraphState) -> GraphState:
    """KnowledgeInquiryAgent 실행"""
    return await knowledge_inquiry_agent.invoke(state, system_prompt=run_knowledge_inquiry.prompt)


# ─── 라우팅 ───────────────────────────────────────────────────

DOMAIN_NODE_FUNCS = {
    knowledge_inquiry_agent.intent_id: (run_knowledge_inquiry, knowledge_inquiry_agent.intent_name),
    analyze_log_agent.intent_id: (run_analyze_log, analyze_log_agent.intent_name),
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
    name="router",
    description="분류된 intent 기준으로 Agent Routing",
    start_node="classifier",
    condition_func=route_by_intent,
    mapping={node_name: node_name for node_name in list(DOMAIN_NODE_FUNCS.keys()) + ["default_response"]},
)

for node_name in DOMAIN_NODE_FUNCS:
    manager.add_edge(node_name, "responser")
manager.add_edge("default_response", "responser")
manager.add_edge("responser", GAIA_STANDARD_OUTPUT_NODE)

graph = manager.compile()
logger.info(f"Agent Workflow 컴파일 완료 (v{VERSION})")
