"""
돌발 질문 서비스.

질문 생성: GPT-4.1-mini
답변 평가 (배치 분석 시):
  내용 점수 (0-100) =
    GPT 적절성     40%  — 질문에 대한 답인가?
    토픽 유사도    20%  — KoSimCSE: 질문 ↔ 답변 코사인 유사도
    문장 품질      20%  — RoBERTa: 자연스러운 말인지, 말이 되는 말인지
    문장 흐름      20%  — 인접 문장 coherence: 말이 잘 이어지는지
"""

import json
import asyncio
import os

from openai import OpenAI
from app.llm.surprise_question_prompts import build_question_prompt, build_eval_prompt

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# KoSimCSE 코사인 유사도 → 0~100 정규화 기준
_SIM_MIN = 0.10
_SIM_MAX = 0.85


def _sim_to_score(cosine: float) -> int:
    norm = max(0.0, min(1.0, (cosine - _SIM_MIN) / (_SIM_MAX - _SIM_MIN)))
    return int(round(norm * 100))


def _quality_to_score(q_raw: float) -> int:
    """quality_score (0~2) → 0~100."""
    return int(min(100, round(q_raw * 50)))


def _coherence_from_analysis(sent_analysis: list) -> int:
    """
    sentence_analysis 결과에서 coherence_sim 평균 추출 → 0~100.
    문장이 1개이면 quality_score로 대체.
    """
    sims = [s["coherence_sim"] for s in sent_analysis if s.get("coherence_sim") is not None]
    if not sims:
        return 50
    avg = sum(sims) / len(sims)
    return int(min(100, max(0, round(avg * 100))))


# ── 질문 생성 ─────────────────────────────────────────────────────────────────

async def generate_question(context_text: str, asked_questions: list[str] | None = None) -> dict | None:
    """
    GPT-4.1-mini로 돌발 질문 1개 생성.
    반환: {"question": str}  실패 시 None
    """
    if not context_text or not context_text.strip():
        return None
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: _client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": build_question_prompt(context_text, asked_questions)}],
            ),
        )
        data = json.loads(res.output_text.strip())
        question = data.get("question", "").strip()
        return {"question": question} if question else None
    except Exception as e:
        print(f"[SurpriseQ] 질문 생성 실패: {e}")
        return None


# ── 답변 평가 (배치 분석 시 호출) ─────────────────────────────────────────────

async def evaluate_answer(
    question: str,
    answer_text: str,
    context_text: str,
    scorer,
) -> dict:
    """
    배치 분석 시 답변 평가.

    Args:
        question:     돌발 질문 텍스트
        answer_text:  질문 이후 슬라이싱된 STT 텍스트
        context_text: 질문 생성 당시 발표 맥락
        scorer:       ServiceScorer 인스턴스

    Returns:
        {
          answered, content_score,
          gpt_score, similarity_score, quality_score, coherence_score,
          feedback
        }
    """
    # 답변 텍스트가 없거나 너무 짧으면 0점
    if not answer_text or len(answer_text.split()) < 5:
        return {
            "answered":        False,
            "content_score":   0,
            "gpt_score":       0,
            "similarity_score": 0,
            "quality_score":   0,
            "coherence_score": 0,
            "feedback":        "질문 이후 발화 내용이 충분하지 않아 평가할 수 없습니다.",
        }

    # ── 1. GPT: 답변 포함 여부 + 적절성 점수 ──────────────────────────────────
    gpt_answered = False
    gpt_score = 0
    gpt_feedback = "평가를 진행할 수 없었습니다."
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: _client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": build_eval_prompt(question, answer_text, context_text),
                }],
            ),
        )
        parsed = json.loads(res.output_text.strip())
        gpt_answered = bool(parsed.get("answered", False))
        gpt_score    = int(parsed.get("score", 0))
        gpt_feedback = parsed.get("feedback", gpt_feedback)
    except Exception as e:
        print(f"[SurpriseQ] GPT 평가 실패: {e}")

    # GPT가 답변 없다고 판단하면 나머지 점수도 0
    if not gpt_answered:
        return {
            "answered":         False,
            "content_score":    0,
            "gpt_score":        0,
            "similarity_score": 0,
            "quality_score":    0,
            "coherence_score":  0,
            "feedback":         gpt_feedback,
        }

    # ── 2. 토픽 모델: 질문 ↔ 답변 코사인 유사도 ───────────────────────────────
    similarity_score = 50
    try:
        loop = asyncio.get_event_loop()
        cosine = await loop.run_in_executor(
            None,
            lambda: scorer.topic_similarity(question, answer_text),
        )
        similarity_score = _sim_to_score(float(cosine))
    except Exception as e:
        print(f"[SurpriseQ] 유사도 계산 실패: {e}")

    # ── 3. RoBERTa: 문장 품질 (자연스러운 말인지) ─────────────────────────────
    quality_score = 50
    try:
        loop = asyncio.get_event_loop()
        q_raw = await loop.run_in_executor(
            None,
            lambda: scorer.quality_score(answer_text),
        )
        quality_score = _quality_to_score(float(q_raw))
    except Exception as e:
        print(f"[SurpriseQ] 품질 점수 계산 실패: {e}")

    # ── 4. 문장 흐름 (coherence: 말이 잘 이어지는지) ─────────────────────────
    coherence_score = 50
    try:
        loop = asyncio.get_event_loop()
        sent_analysis = await loop.run_in_executor(
            None,
            lambda: scorer.sentence_analysis(question, answer_text),
        )
        coherence_score = _coherence_from_analysis(sent_analysis)
    except Exception as e:
        print(f"[SurpriseQ] 흐름 점수 계산 실패: {e}")

    # ── 5. 최종 내용 점수: GPT 40% + 유사도 20% + 품질 20% + 흐름 20% ─────────
    content_score = int(round(
        gpt_score       * 0.4 +
        similarity_score * 0.2 +
        quality_score    * 0.2 +
        coherence_score  * 0.2
    ))

    return {
        "answered":         True,
        "content_score":    content_score,
        "gpt_score":        gpt_score,
        "similarity_score": similarity_score,
        "quality_score":    quality_score,
        "coherence_score":  coherence_score,
        "feedback":         gpt_feedback,
    }
