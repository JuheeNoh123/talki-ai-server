def build_question_prompt(context_text: str, asked_questions: list[str] | None = None) -> str:
    asked_section = ""
    if asked_questions:
        asked_list = "\n".join(f"- {q}" for q in asked_questions)
        asked_section = f"\n[이미 출제된 질문 — 반드시 제외]\n{asked_list}\n"

    return (
        "너는 발표 코칭 전문가다.\n"
        "발표자가 방금까지 발표한 내용을 바탕으로, 발표자의 이해도와 설명력을 확인할 수 있는 돌발 질문 1개를 만들어라.\n\n"
        f"[발표 내용 (최근 발화)]\n{context_text}\n"
        f"{asked_section}\n"
        "[규칙]\n"
        "- 질문은 1개만 생성\n"
        "- 발표 내용에서 핵심 개념이나 주요 주장과 관련된 질문\n"
        "- 발표자가 30초 이내에 구두로 답할 수 있는 수준\n"
        "- 짧고 명확하게 한국어로 작성 (1~2문장)\n"
        "- 이미 출제된 질문과 동일하거나 유사한 질문 금지\n"
        "- 발표 내용에 명확히 등장하는 개념·주장만 질문할 것\n"
        "- 의미를 알 수 없거나 문맥상 어색한 단어·문장(STT 오인식 가능성)은 절대 인용하거나 질문 소재로 삼지 말 것\n"
        "- 발표 전체 흐름에서 자연스럽게 이해되는 내용만 다룰 것\n"
        '- 반드시 JSON 형식으로만 출력: {"question": "..."}'
    )


def build_eval_prompt(question: str, answer_text: str, context_text: str) -> str:
    answer_display = answer_text.strip() if answer_text and answer_text.strip() else "(텍스트 없음)"
    return (
        "너는 발표 코칭 전문가다.\n"
        "발표자에게 돌발 질문을 했고, 질문 이후 발표자가 말한 내용이 아래에 있다.\n"
        "이 텍스트에 질문에 대한 답변이 포함되어 있는지 판단하고, 있다면 얼마나 적절한지 평가하라.\n\n"
        f"[발표 맥락]\n{context_text}\n\n"
        f"[돌발 질문]\n{question}\n\n"
        f"[질문 이후 발화 내용]\n{answer_display}\n\n"
        "[평가 기준]\n"
        "1. 질문에 대한 답변이 이 텍스트에 포함되어 있는가? (발표 진행인지 답변인지 구분)\n"
        "2. 답변이 있다면: 질문을 제대로 이해하고 답했는가?\n"
        "3. 답변이 있다면: 발표 맥락과 일치하고 내용이 정확한가?\n\n"
        "[규칙]\n"
        "- answered: 답변이 포함되어 있으면 true, 그냥 발표 이어나간 것이면 false\n"
        "- score: answered=true이면 0~100, false이면 반드시 0\n"
        "- feedback: 2~3문장 한국어\n"
        '- 반드시 JSON 형식으로만 출력: {"answered": true/false, "score": 숫자, "feedback": "..."}'
    )
