# app/llm/practice_prompts.py
# 연습 탭(즉흥 구성 연습) LLM 생성/평가용 프롬프트 빌더.
# 호출 패턴은 app/services/surprise_question_service.py의 build_question_prompt/build_eval_prompt와 동일하게
# "단일 user 메시지 + JSON 강제" 형태를 따른다.


def build_impromptu_topic_prompt(asked_questions: list[str] | None = None) -> str:
    asked_section = ""
    if asked_questions:
        asked_list = "\n".join(f"- {q}" for q in asked_questions)
        asked_section = f"[이미 출제된 질문 — 반드시 제외]\n{asked_list}\n\n"

    return (
        "너는 발표 불안 완화 훈련을 설계하는 코치다.\n"
        "사용자가 10초 준비 후 30초간 즉흥으로 답변할 수 있는, 부담스럽지 않고 일상적인 질문을 1개 생성하라.\n\n"
        f"{asked_section}"
        "[조건]\n"
        "- 개인 경험이나 의견을 묻는 형태(예: \"최근에 ~한 경험은?\", \"~에 대해 어떻게 생각하나요?\")\n"
        "- 논쟁적이거나 민감한 주제(정치/종교/개인정보) 금지\n"
        "- 30초 안에 답변 가능한 난이도\n"
        "- 짧고 명확하게 한국어 1문장으로 작성\n"
        '- 반드시 JSON 형식으로만 출력: {"question": "..."}'
    )


def build_keyword_prompt() -> str:
    return (
        "너는 발표 구성력 훈련을 설계하는 코치다.\n"
        "서로 자연스럽게 하나의 이야기로 연결할 수 있는 한국어 키워드 3개를 생성하라.\n\n"
        "[조건]\n"
        "- 키워드는 명사형 단어 2~5글자\n"
        "- 지나치게 추상적이거나 서로 무관한 조합 금지 (예: \"우주\"+\"김치\"+\"회계\")\n"
        "- 직장/학교/일상 경험과 자연스럽게 엮일 수 있는 소재 우선\n"
        '- 반드시 JSON 형식으로만 출력: {"keywords": ["", "", ""]}'
    )


def build_gist_prompt() -> str:
    return (
        "너는 발표 핵심 요약 훈련을 설계하는 코치다.\n"
        "200~300자 분량의 한국어 글 1개와, 그 글의 핵심을 담은 참고 키워드 3개를 생성하라.\n\n"
        "[조건]\n"
        "- 글은 2문단 이내, 팀워크/커뮤니케이션/자기계발 등 무난한 소재\n"
        "- 참고 키워드 3개는 글의 핵심 메시지를 요약했을 때 반드시 언급되어야 하는 단어\n"
        '- 반드시 JSON 형식으로만 출력: {"source_text": "", "reference_keywords": ["", "", ""]}'
    )


def build_keyword_eval_prompt(keywords: list[str], text: str) -> str:
    return (
        "너는 말하기 연습 코치다. 아래 발화가 주어진 키워드들을 얼마나 자연스럽게 연결했는지 평가하라.\n\n"
        f"[키워드]\n{keywords}\n\n"
        f"[발화 텍스트]\n{text}\n\n"
        "[규칙]\n"
        '- naturalness_label: "우수"|"양호"|"보통"|"미흡" 중 하나 (키워드 간 연결이 억지스럽지 않은 정도)\n'
        '- 반드시 JSON 형식으로만 출력: {"naturalness_label": ""}'
    )


def build_gist_eval_prompt(reference_keywords: list[str], source_text: str, spoken_text: str) -> str:
    return (
        "너는 말하기 연습 코치다. 아래 원문의 핵심을 발화자가 얼마나 정확히 요약했는지 평가하라.\n\n"
        f"[원문]\n{source_text}\n\n"
        f"[참고 키워드]\n{reference_keywords}\n\n"
        f"[발화자의 요약 텍스트]\n{spoken_text}\n\n"
        "[규칙]\n"
        '- gist_accuracy_label: "우수"|"양호"|"보통"|"미흡" 중 하나\n'
        "- missing_points: 발화에서 빠진 핵심 키워드 목록 (배열, 없으면 빈 배열)\n"
        '- 반드시 JSON 형식으로만 출력: {"gist_accuracy_label": "", "missing_points": []}'
    )


def build_structure_eval_prompt(topic: str, spoken_text: str) -> str:
    return (
        "너는 말하기 연습 코치다. 아래는 즉흥 발표 주제와 발화자의 답변이다. 답변의 구성 완성도를 평가하라.\n\n"
        f"[주제]\n{topic}\n\n"
        f"[답변 텍스트]\n{spoken_text}\n\n"
        "[규칙]\n"
        "- 질문에 적절히 답했는지, 서두-본론-마무리 흐름이 있는지를 고려\n"
        '- structure_completeness: "우수"|"양호"|"보통"|"미흡" 중 하나\n'
        '- 반드시 JSON 형식으로만 출력: {"structure_completeness": ""}'
    )
