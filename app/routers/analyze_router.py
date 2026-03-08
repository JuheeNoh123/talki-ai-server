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

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/record")
async def analyze_record(
    # presentation_type: str = Query(
    #     PresentationType.ONLINE_SMALL,
    #     description="발표 유형 (online_small | small | large)"
    # ),
    # file: UploadFile = File(...)
    req: AnalyzeFromS3Request
):
    #loop = asyncio.get_running_loop()
    #print("ANALYZE RECORD FROM S3 HIT")
    #print("FILE =", file.filename)
    # video_path = f"temp_{file.filename}"
    # with open(video_path, "wb") as f:
    #     f.write(await file.read())
    # raw_result = await loop.run_in_executor(
    #     None,  # ThreadPoolExecutor
    #     analyzer.analyze_record_video,
    #     video_path
    # )
    # feedback = feedback_service.generate_feedback(raw_result, presentation_type)

    # return {
    #     "raw_result": raw_result,
    #     "feedback": feedback
    # }

    asyncio.create_task(background_analysis(req))

    return {"status": "processing"}

   

async def background_analysis(req):

    async with analysis_semaphore:  # 🔥 동시 실행 제한

        print("🚀 분석 시작")
        
        video_path = download_video(req.video_url)

        loop = asyncio.get_running_loop()

        try:
            raw_result = await loop.run_in_executor(
                None,
                analyzer.analyze_record_video,
                video_path
            )

            feedback = feedback_service.generate_feedback(
                raw_result,
                req.presentation_type
            )

            requests.post(
                "http://localhost:8000/analyze/callback",
                json={
                    "s3_key": req.s3_key,
                    "raw_result": raw_result,
                    "feedback": feedback
                }
            )

        finally:
            import os
            os.remove(video_path)
            print("✅ 분석 종료")
