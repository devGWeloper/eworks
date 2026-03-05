"""회사 내부 config 라이브러리 대체 모듈 (임시)

이관 시 이 파일을 삭제하고, import를 회사 내부 라이브러리로 교체한다.
회사에서는 config.dev.yml 파일에서 설정값을 읽어온다.
"""

from pathlib import Path

import yaml

_config_path = Path(__file__).parents[4] / "config.dev.yml"
with open(_config_path, encoding="utf-8") as _f:
    _data: dict = yaml.safe_load(_f) or {}

# ── Service ──
SERVICE_ID: str = str(_data.get("SERVICE_ID", ""))

# ── Embedding (DOCU) ──
DOCU_EMBEDDING_MODEL_NAME: str = str(_data.get("DOCU_EMBEDDING_MODEL_NAME", ""))
DOCU_EMBEDDING_ENDPOINT: str = str(_data.get("DOCU_EMBEDDING_ENDPOINT", ""))
DOCU_EMBEDDING_API_KEY: str = str(_data.get("DOCU_EMBEDDING_API_KEY", ""))
DOCU_EMBEDDING_CTX_LENGTH: int = int(_data.get("DOCU_EMBEDDING_CTX_LENGTH", 8191))

# ── Collection (DOCU / KNOWHOW) ──
DOCU_USER_COLLECTION_NAME: str = str(_data.get("DOCU_USER_COLLECTION_NAME", ""))
DOCU_VECTOR_DB_URI: str = str(_data.get("DOCU_VECTOR_DB_URI", ""))
KNOWHOW_USER_COLLECTION_NAME: str = str(_data.get("KNOWHOW_USER_COLLECTION_NAME", ""))
KNOWHOW_VECTOR_DB_URI: str = str(_data.get("KNOWHOW_VECTOR_DB_URI", ""))

# ── LLM ──
PRIVATE_LLM_MODEL_NAME: str = str(_data.get("PRIVATE_LLM_MODEL_NAME", ""))
PRIVATE_LLM_ENDPOINT: str = str(_data.get("PRIVATE_LLM_ENDPOINT", ""))
PRIVATE_LLM_API_KEY: str = str(_data.get("PRIVATE_LLM_API_KEY", ""))

# ── Tool (MCP) ──
MCP_SERVER_HOST: str = str(_data.get("MCP_SERVER_HOST", "localhost"))
MCP_SERVER_PORT: str = str(_data.get("MCP_SERVER_PORT", "8080"))
