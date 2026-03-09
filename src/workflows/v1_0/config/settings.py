"""Agent 설정"""

from dataclasses import dataclass
from enum import Enum

from ..temp import config

# ── Config Dataclass ──


@dataclass
class EmbeddingConfig:
    """Embedding 모델 설정"""

    model: str
    api_key: str = ""


@dataclass
class CollectionConfig:
    """벡터 DB Collection 설정"""

    collection_name: str
    uri: str
    embedding: str
    llm: str = ""  # 비어있으면 DEFAULT_LLM_MODEL 사용
    token: str = ""
    vector_field: str = "vector"
    vector_fields: dict = None  # 다중 dense 필드: {"title_dense": 0.6, "text_dense": 0.4}
    text_field: str = "text"
    sparse_vector_field: str = ""  # 설정 시 hybrid search 활성화


@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""

    model: str
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.1


@dataclass
class MCPServerConfig:
    """MCP Server 연결 설정"""

    url: str
    transport: str = "sse"
    timeout: int = 30


# ── LLM 모델 ──


class LLMModel(str, Enum):
    """등록된 LLM 모델 이름"""

    QWEN3 = "Qwen3-235B-A22B-Instruct-2507-AWQ"


LLM_MODELS = {
    LLMModel.QWEN3: LLMModelConfig(
        model=config.PRIVATE_LLM_MODEL_NAME,
        base_url=config.PRIVATE_LLM_ENDPOINT,
        api_key=config.PRIVATE_LLM_API_KEY,
    ),
}

DEFAULT_LLM_MODEL = LLMModel.QWEN3

# ── Retrieval ──


class Embedding(str, Enum):
    """등록된 Embedding 모델 이름"""

    BGE_M3 = "bge_m3"


EMBEDDINGS = {
    Embedding.BGE_M3: EmbeddingConfig(
        model=config.DOCU_EMBEDDING_MODEL_NAME,
        api_key=config.DOCU_EMBEDDING_API_KEY,
    ),
}


class Collection(str, Enum):
    """등록된 Collection 이름"""

    KNOWHOW = "knowhow"
    UPLOAD = "upload"


RETRIEVAL_COLLECTIONS = {
    Collection.KNOWHOW: CollectionConfig(
        collection_name=config.KNOWHOW_USER_COLLECTION_NAME,
        uri=config.KNOWHOW_VECTOR_DB_URI,
        embedding=Embedding.BGE_M3,
        llm=LLMModel.QWEN3,
        vector_field="knowhow_embedded_vector",
        text_field="knowhow",
    ),
    Collection.UPLOAD: CollectionConfig(
        collection_name=config.DOCU_USER_COLLECTION_NAME,
        uri=config.DOCU_VECTOR_DB_URI,
        embedding=Embedding.BGE_M3,
        llm=LLMModel.QWEN3,
        vector_field="dense_vector",
        sparse_vector_field="sparse_vector",
        text_field="text",
    ),
}

# ── Tool ──


class MCPServer(str, Enum):
    """등록된 MCP Server 이름"""

    MCP_SERVER = "mcp_server"


MCP_SERVERS = {
    MCPServer.MCP_SERVER: MCPServerConfig(
        url=f"http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}/sse",
    ),
}
