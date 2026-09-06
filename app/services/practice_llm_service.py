# app/services/practice_llm_service.py
# 연습 탭(즉흥 구성 연습) 주제/키워드/원문 생성 및 발화 정성 평가.
# 호출 패턴은 app/services/surprise_question_service.py를 그대로 따른다
# (OpenAI client.responses.create, run_in_executor로 블로킹 호출, JSON 파싱).

import asyncio
import json
import os

from openai import OpenAI

from app.llm.practice_prompts import (
    build_gist_eval_prompt,
    build_gist_prompt,
    build_impromptu_topic_prompt,
    build_keyword_eval_prompt,
    build_keyword_prompt,
    build_structure_eval_prompt,
)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_FALLBACK_TOPIC = "최근에 배운 것 중 가장 유용했던 것은 무엇인가요?"
_FALLBACK_KEYWORDS = ["협업", "문제해결", "성장"]
_FALLBACK_PASSAGE = (
    "효과적인 팀워크를 위해서는 무엇보다도 명확하고 원활한 의사소통이 필수적입니다. "
    "단순히 정보를 전달하는 데 그치는 것이 아니라, 서로의 생각과 의도를 정확히 이해하고 "
    "공감하는 과정이 함께 이루어져야 합니다. 또한 팀워크는 개인의 역량만으로 완성되는 것이 "
    "아니라, 공동의 목표를 향해 함께 나아가려는 협력 정신 속에서 더욱 빛을 발합니다."
)
_FALLBACK_REFERENCE_KEYWORDS = ["의사소통", "경청", "협력"]


async def _call_llm(prompt: str) -> dict | None:
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: _client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": prompt}],
            ),
        )
        return json.loads(res.output_text.strip())
    except Exception as e:
        print(f"[PracticeLLM] 호출 실패: {e}")
        return None


async def generate_impromptu_topic(asked_questions: list[str] | None = None) -> dict:
    data = await _call_llm(build_impromptu_topic_prompt(asked_questions))
    question = (data or {}).get("question", "").strip()
    return {"question": question or _FALLBACK_TOPIC}


async def generate_keywords() -> dict:
    data = await _call_llm(build_keyword_prompt())
    keywords = (data or {}).get("keywords")
    if not keywords or len(keywords) < 3:
        keywords = _FALLBACK_KEYWORDS
    return {"keywords": list(keywords)[:3]}


async def generate_gist_passage() -> dict:
    data = await _call_llm(build_gist_prompt())
    source_text = (data or {}).get("source_text", "").strip()
    reference_keywords = (data or {}).get("reference_keywords")
    if not source_text:
        source_text = _FALLBACK_PASSAGE
    if not reference_keywords or len(reference_keywords) < 3:
        reference_keywords = _FALLBACK_REFERENCE_KEYWORDS
    return {"source_text": source_text, "reference_keywords": list(reference_keywords)[:3]}


async def evaluate_keyword_connection(keywords: list[str], text: str) -> dict:
    data = await _call_llm(build_keyword_eval_prompt(keywords, text))
    if not data:
        return {"naturalness_label": "보통"}
    return {"naturalness_label": data.get("naturalness_label", "보통")}


async def evaluate_gist_accuracy(reference_keywords: list[str], source_text: str, spoken_text: str) -> dict:
    data = await _call_llm(build_gist_eval_prompt(reference_keywords, source_text, spoken_text))
    if not data:
        return {"gist_accuracy_label": "보통", "missing_points": []}
    return {
        "gist_accuracy_label": data.get("gist_accuracy_label", "보통"),
        "missing_points": data.get("missing_points", []),
    }


async def evaluate_structure_completeness(topic: str, spoken_text: str) -> dict:
    data = await _call_llm(build_structure_eval_prompt(topic, spoken_text))
    if not data:
        return {"structure_completeness": "보통"}
    return {"structure_completeness": data.get("structure_completeness", "보통")}
