"""AnalyzeLog 프롬프트 템플릿

바인딩 변수:
  - target: 확인 대상
  - condition: 확인 조건
"""

# TODO: 실제 존재 여부 확인 구현 시 프롬프트 확정
ANALYZE_LOG_PROMPT = """다음 조건의 존재 여부를 확인합니다.

## 확인 대상
{target}

## 확인 조건
{condition}"""
