"""AnalyzeLog 프롬프트 템플릿

SYSTEM: Agent identity + 지시사항
USER: 확인 대상 + 확인 조건

바인딩 변수 (system):
  - agent_prompt: Agent identity 프롬프트
바인딩 변수 (user):
  - target: 확인 대상
  - condition: 확인 조건
"""

# TODO: 실제 존재 여부 확인 구현 시 프롬프트 확정
ANALYZE_LOG_SYSTEM_PROMPT = """$agent_prompt

## 지시사항
사용자가 요청한 대상의 존재 여부를 모니터링 데이터에서 확인하세요.
Tool을 활용하여 데이터를 조회하고, 결과를 종합하여 응답하세요."""

ANALYZE_LOG_USER_PROMPT = """다음 조건의 존재 여부를 확인해줘.

## 확인 대상
$target

## 확인 조건
$condition"""
