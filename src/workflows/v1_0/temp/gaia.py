"""회사 내부 GAIA 라이브러리 대체 모듈 (임시)

이관 시 이 파일을 삭제하고, import를 회사 내부 라이브러리로 교체한다.
"""

from typing import Callable, Type

from langgraph.graph import END, StateGraph

GAIA_OUTPUT_END = END
GAIA_STANDARD_OUTPUT_NODE = END


# ── AgentManager ──


class AgentManager(StateGraph):
    """StateGraph 확장 - 워크플로우 메타데이터와 스트리밍 기능 제공"""

    def __init__(
        self,
        state_schema: Type,
        service_id: str = "",
        author: str = "",
        workflow_name: str = "",
        description: str = "",
        workflow_version: str = "1.0",
        main_model: str = "",
    ):
        super().__init__(state_schema)
        self.service_id = service_id
        self.author = author
        self.workflow_name = workflow_name
        self.description = description
        self.workflow_version = workflow_version
        self.main_model = main_model

    def add_node(
        self,
        name: str,
        func: Callable,
        *,
        description: str = "",
        **kwargs,
    ) -> None:
        """노드 등록 (description은 metadata에 저장)"""
        metadata = kwargs.pop("metadata", {})
        if description:
            metadata["description"] = description
        super().add_node(name, func, metadata=metadata, **kwargs)

    def add_conditional_edges(
        self,
        start_node: str,
        condition_func: Callable,
        mapping: dict,
        *,
        name: str = "",
        description: str = "",
        **kwargs,
    ) -> None:
        """조건부 엣지 등록 (회사 시그니처 호환)"""
        super().add_conditional_edges(start_node, condition_func, mapping, **kwargs)

    def middle_stream_text(self, text: str, name: str = "", description: str = "") -> bool:
        """중간 스트리밍 텍스트 처리"""
        # TODO: 회사 라이브러리의 스트리밍 메커니즘에 맞게 구현
        print(f"[Stream] {name}: {text}")
        return True

    @staticmethod
    def main_model_type():
        """메인 모델 타입 데코레이터 (회사 시그니처 호환, no-op)"""

        def decorator(func):
            return func

        return decorator

    @staticmethod
    def prompt_type(text: str):
        """함수에 프롬프트 템플릿을 바인딩하는 데코레이터

        사용:
            @manager.prompt_type(NORMALIZER_PROMPT)
            async def run_normalizer(state):
                ...

            run_normalizer.prompt  # → NORMALIZER_PROMPT 문자열
        """

        def decorator(func):
            func.prompt = text
            return func

        return decorator
