"""Retrieval 예외 정의"""


class RetrievalConfigError(Exception):
    """Retrieval 설정 오류 — Collection/Embedding 미등록 또는 매핑 누락"""

    def __init__(self, message: str):
        super().__init__(message)


class RetrievalConnectionError(Exception):
    """벡터 DB 연결 실패 — 재시도 소진 후 발생"""

    def __init__(self, collection_name: str, cause: Exception):
        self.collection_name = collection_name
        self.cause = cause
        super().__init__(f"Collection '{collection_name}' 연결 실패: {cause}")
