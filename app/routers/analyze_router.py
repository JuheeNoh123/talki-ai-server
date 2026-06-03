# app/routers/analyze_router.py
from fastapi import APIRouter, UploadFile, File, Query
from app.services import analyze_service_optimized as analyzer
from app.services import feedback_service
import cv2
import numpy as np
import asyncio
analysis_semaphore = asyncio.Semaphore(2)
from app.config.feedback_criteria import PresentationType
from app.schemas.analyze_schema import AnalyzeFromS3Request
import requests
import tempfile
import os
import sys
from pathlib import Path

TOPIC_MODEL_DIR = os.getenv(
    "TOPIC_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "Topic_model")
)
if TOPIC_MODEL_DIR not in sys.path:
    sys.path.insert(0, TOPIC_MODEL_DIR)

SPRING_URL = os.getenv("SPRING_URL")

from app.core.redis import redis_client
from app.core.scorer import scorer
from app.services import surprise_question_service
import json

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/record-from-s3")
async def analyze_record(
    # presentation_type: str = Query(
    #     PresentationType.ONLINE_SMALL,
    #     description="발표 유형 (online_small | small | large)"
    # ),
    # file: UploadFile = File(...)
    req: AnalyzeFromS3Request
):
    #print(req)
    asyncio.create_task(background_analysis(req))

    return {"status": "processing"}

def download_video(url: str) -> str:
    print("📥 영상 다운로드 시작")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 임시 파일 생성
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    # 큰 파일 대비 chunk 단위로 저장
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)

    tmp.close()

    print("✅ 다운로드 완료:", tmp.name)

    return tmp.name

async def background_analysis(req):

    async with analysis_semaphore:  # 🔥 동시 실행 제한
        #print(req.dict())

        print("🚀 분석 시작")
        
        video_path = download_video(req.video_url)

        loop = asyncio.get_running_loop()

        try:
            raw_result = await loop.run_in_executor(
                None,
                analyzer.analyze_record_video,
                video_path
            )
            #print(raw_result)
            text = raw_result.get("stt_text", "") #텍스트 추출
            #print("stt 텍스트",text)

            # 🔥 주제 분석
            topic_result = scorer.predict(
                topic_summary=req.topic_summary,
                topic_desc=req.topic_desc,
                topic_tags=req.topic_tags,
                doc_text=text
            )

            raw_result["topic"] = topic_result

            # 배치 분석 시선 Y축 플립 (실시간 analyze_service_landmarks와 동일 방향으로 통일)
            eyes = raw_result.get("eyes", {})
            if eyes:
                old_vert_mode   = eyes.get("vert_mode", "center")
                old_vert_counts = eyes.get("vert_counts", {})
                raw_result["eyes"]["avg_dy"]     = -eyes.get("avg_dy", 0)
                raw_result["eyes"]["vert_mode"]  = (
                    "down" if old_vert_mode == "up" else
                    "up"   if old_vert_mode == "down" else "center"
                )
                raw_result["eyes"]["vert_counts"] = {
                    "up":     old_vert_counts.get("down", 0),
                    "center": old_vert_counts.get("center", 0),
                    "down":   old_vert_counts.get("up", 0),
                }

            # 돌발 질문 평가 (generate_feedback 전에 수행 → llm_feedback에 포함)
            surprise_questions = []
            if req.presentation_id:
                try:
                    pending_list = await redis_client.lrange(
                        f"presentation:{req.presentation_id}:surprise_questions_pending", 0, -1
                    )
                    print(f"[SurpriseQ] presentation_id: {req.presentation_id}")
                    print(f"[SurpriseQ] 대기 중인 질문 수: {len(pending_list)}")

                    wpm       = raw_result.get("WPM", 100) or 100
                    stt_words = raw_result.get("stt_text", "").split()

                    for pending_json in pending_list:
                        pending  = json.loads(pending_json)
                        asked_at = pending.get("asked_at_seconds", 0)
                        word_pos = min(int(asked_at * wpm / 60), len(stt_words))
                        answer_text = " ".join(stt_words[word_pos: word_pos + 80])

                        print(f"[SurpriseQ] 평가 중 — {pending['question'][:30]}...")
                        result = await surprise_question_service.evaluate_answer(
                            question=pending["question"],
                            answer_text=answer_text,
                            context_text=pending.get("context_text", ""),
                            scorer=scorer,
                        )
                        surprise_questions.append({
                            "question_id":      pending["question_id"],
                            "question":         pending["question"],
                            "asked_at_seconds": asked_at,
                            "answer_text":      answer_text,
                            **result,
                        })
                    print(f"[SurpriseQ] 평가 완료 — 총 {len(surprise_questions)}개")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[SurpriseQ] 돌발 질문 평가 실패: {e}")

            feedback = feedback_service.generate_feedback(
                raw_result,
                req.presentation_type,
                surprise_questions,
            )

            gaze = raw_result.get("eyes", {})
            raw_data = {
                "speech": {
                    "wpm": raw_result.get("WPM", 0),
                    "fillers_count": raw_result.get("fillers_count", 0),
                    "fillers_freq": raw_result.get("fillers_freq", 0.0),
                    "filler_detail": raw_result.get("filler_detail", {}),
                    "filler_list": raw_result.get("filler_list", []),
                    "silence_count": raw_result.get("silence_count", 0),
                    "total_silence_sec": raw_result.get("total_silence_sec", 0.0),
                    "silence_ratio": raw_result.get("silence_ratio", 0.0),
                    "text": raw_result.get("stt_text", ""),
                },
                "pose": {
                    "avg_speed": raw_result.get("handArmMovementAvg", 0.0),
                    "max_speed": raw_result.get("handArmMovementMaxRolling", 0.0),
                    "warning_count": raw_result.get("pose_warning_count", 0),
                    "warning_ratio": raw_result.get("pose_warning_ratio", 0.0),
                    "rigid_count": raw_result.get("pose_rigid_count", 0),
                    "rigid_ratio": raw_result.get("pose_rigid_ratio", 0.0),
                    "samples": raw_result.get("pose_samples", 0),
                },
                "gaze": {
                    "avg_dx": gaze.get("avg_dx", 0.0),
                    "avg_dy": gaze.get("avg_dy", 0.0),
                    "horizontal_mode": gaze.get("horiz_mode", ""),
                    "vertical_mode": gaze.get("vert_mode", ""),
                    "horizontal_counts": gaze.get("horiz_counts", {}),
                    "vertical_counts": gaze.get("vert_counts", {}),
                    "samples": gaze.get("samples", 0),
                    "horiz_front_ratio": round(
                        gaze.get("horiz_counts", {}).get("center", 0) / gaze.get("samples", 1)
                        if gaze.get("samples", 0) > 0 else 0.0, 3
                    ),
                    "vert_front_ratio": round(
                        gaze.get("vert_counts", {}).get("center", 0) / gaze.get("samples", 1)
                        if gaze.get("samples", 0) > 0 else 0.0, 3
                    ),
                    # front_ratio: scoring과 동일한 가중 평균 (presentation_type별 가중치 적용)
                    "front_ratio": round(
                        (lambda s=gaze.get("samples", 0),
                                 hc=gaze.get("horiz_counts", {}).get("center", 0),
                                 vc=gaze.get("vert_counts", {}).get("center", 0),
                                 pt=req.presentation_type:
                            (hc / s) * {"online_small": 0.5, "small": 0.4, "large": 0.6}.get(pt, 0.5) +
                            (vc / s) * {"online_small": 0.5, "small": 0.6, "large": 0.4}.get(pt, 0.5)
                            if s > 0 else 0.0
                        )(), 3
                    ),
                },
                "topic": raw_result.get("topic", {}),
                "surprise_questions": surprise_questions,
            }

            surprise_score = (
                round(sum(q["content_score"] for q in surprise_questions) / len(surprise_questions))
                if surprise_questions else None
            )

            resjson = {
                "s3_key": req.s3_key,
                "raw_data": raw_data,
                "scores": {
                    "total_score": feedback["total_score"],
                    "score_detail": {
                        **feedback["score_detail"],
                        "surprise": surprise_score,
                    },
                },
                "llm_feedback": feedback["llm_feedback"],
                "surprise_questions": surprise_questions,
            }

            requests.post(
                f"{SPRING_URL}/analyze/callback",
                json=resjson
            )

            print(resjson)
            # print("callback:",req.s3_key)

        finally:
            import os
            os.remove(video_path)
            print("✅ 분석 종료")
