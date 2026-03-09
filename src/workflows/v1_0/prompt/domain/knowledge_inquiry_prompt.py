"""KnowledgeInquiry 프롬프트 템플릿

SYSTEM: Agent identity + 지시사항
USER: 조회 대상 + Agent intent 카탈로그

바인딩 변수 (system):
  - agent_prompt: Agent identity 프롬프트
바인딩 변수 (user):
  - topic: 조회 대상
  - intent_catalog: Agent intent 카탈로그 JSON
"""

KNOWLEDGE_INQUIRY_SYSTEM_PROMPT = """{agent_prompt}

## 지시사항
사용자의 조회 대상에 맞는 정보를 제공하세요.
Agent 기능에 대한 질의인 경우, Agent Intent 카탈로그를 바탕으로 각 기능의 이름, 설명, 사용 예시를 포함하여 안내하세요.
그 외 지식, 메뉴얼, 기준정보 조회인 경우, 해당 주제에 대해 답변하세요."""

KNOWLEDGE_INQUIRY_USER_PROMPT = """## 조회 대상
{topic}

## Agent Intent 카탈로그
{intent_catalog}"""
