# app/routers/websocket_router.py
from fastapi import APIRouter, WebSocket
from app.services import analyze_service_landmarks as analyzer
from app.services.feedback_manager import FeedbackManager
import base64, cv2, numpy as np, json
import time, uuid
from app.core.redis import redis_client



router = APIRouter(tags=["Realtime Analysis"])

@router.websocket("/realtime")
async def realtime_socket(ws: WebSocket):
    """
    실시간 분석 WebSocket
    - frame + audio(base64) 형태로 받음
    - 분석 결과 + 피드백 실시간 전송
    """
    await ws.accept()
    print("[WebSocket] 연결 시작")
    feedback_manager = FeedbackManager()
    presentation_id = uuid.uuid4().hex
    presentation_start_time = time.time()
    await ws.send_text(json.dumps({
        "type": "session_start",
        "presentationId": presentation_id
    }, ensure_ascii=False))

    try:
        while True:
            # 클라이언트로부터 랜드마크 JSON 수신
            # { "face": {...}, "pose": {...}, "timestamp": ... }
            data = await ws.receive_json()

            # 1. 랜드마크 기반 분석
            # (이미지 디코딩 및 미디어파이프 추론 과정 생략 -> 속도 대폭 향상)
            raw_result = analyzer.analyze_realtime_landmarks(data)
            manager_feedback = feedback_manager.update(raw_result)
            # 2. Stateful Feedback Update
            if manager_feedback:
                elapsed_sec = time.time() - presentation_start_time

                event = {
                    "timestamp": round(elapsed_sec, 1),
                    "message": manager_feedback
                }
                # 1️⃣ Redis 저장
                redis_client.rpush(
                    f"presentation:{presentation_id}:feedbacks",
                    json.dumps(event, ensure_ascii=False)
                )
                redis_client.expire(
                    f"presentation:{presentation_id}:feedbacks",
                    60 * 60  # 1시간 TTL
                )

            # 2️⃣ 클라이언트로 전송
            await ws.send_text(json.dumps({
                "type": "feedback",
                "raw_result": raw_result,
                "data": manager_feedback
            }, ensure_ascii=False))

    except Exception as e:
        print(f"[WebSocket] 연결 종료: {e}")
    finally:
        await ws.close()
