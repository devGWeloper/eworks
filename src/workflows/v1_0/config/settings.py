"""Agent 설정"""

import os
from enum import Enum

from core.services.llm_model.llm_model_config import LLMModelConfig
from core.services.retrieval.collection_config import CollectionConfig, EmbeddingConfig
from core.services.tool.mcp_server_config import MCPServerConfig

# ── Retrieval ──


class EmbeddingName(str, Enum):
    """등록된 Embedding 모델 이름 Enum"""

    OPENAI_EMBED = "openai-embed"


class CollectionName(str, Enum):
    """등록된 Collection 이름 Enum"""

    KNOWHOW = "knowhow"


RETRIEVAL_EMBEDDINGS = {
    EmbeddingName.OPENAI_EMBED: EmbeddingConfig(
        model="text-embedding-3-small",
        openai_api_base=os.getenv("OPENAI_EMBED_BASE_URL", ""),
        openai_api_key=os.getenv("OPENAI_EMBED_API_KEY", ""),
    ),
}

RETRIEVAL_COLLECTIONS = {
    CollectionName.KNOWHOW: CollectionConfig(
        collection_name="knowhow",
        uri=os.getenv("MILVUS_URI", ""),
        token=os.getenv("MILVUS_TOKEN", ""),
        embedding_key=EmbeddingName.OPENAI_EMBED,
    ),
}

# ── LLM 모델 ──


class LLMModelName(str, Enum):
    """등록된 LLM 모델 이름 Enum"""

    LLAMA3_70B = "llama-3.3-70b-versatile"


LLM_MODELS = {
    LLMModelName.LLAMA3_70B: LLMModelConfig(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", ""),
    ),
}

DEFAULT_LLM_MODEL = LLMModelName.LLAMA3_70B

# ── Tool ──


class MCPServerName(str, Enum):
    """등록된 MCP Server 이름 Enum"""

    FACTORY = "factory"


MCP_SERVERS = {
    MCPServerName.FACTORY: MCPServerConfig(
        url=os.getenv("MCP_FACTORY_URL", "http://localhost:8080/sse"),
    ),
}
