# app/routers/analyze_router.py
from fastapi import APIRouter, UploadFile, File, Query
from app.services import analyze_service_optimized as analyzer
from app.services import feedback_service
import cv2
import numpy as np
import asyncio
from app.config.feedback_criteria import PresentationType

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/record")
async def analyze_record(
    presentation_type: str = Query(
        PresentationType.ONLINE_SMALL,
        description="발표 유형 (online_small | small | large)"
    ),
    file: UploadFile = File(...)
):
    """녹화 영상 전체 분석 (병렬 처리 + Whisper 1회 로드)
    녹화 영상 분석 API
    - mp4 업로드 시, test_record_multiprocess 기반으로 전체 분석
    - Whisper 병렬 최적화 포함
    - 분석 결과 + 피드백 반
    """
    loop = asyncio.get_running_loop()

    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    raw_result = await loop.run_in_executor(
        None,  # ThreadPoolExecutor
        analyzer.analyze_record_video,
        video_path
    )
    feedback = feedback_service.generate_feedback(raw_result, presentation_type)

    return {
        "raw_result": raw_result,
        "feedback": feedback
    }
