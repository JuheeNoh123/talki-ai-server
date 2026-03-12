# app/routers/websocket_router.py
from fastapi import APIRouter, WebSocket
from app.services import analyze_service_landmarks as analyzer
from app.services.feedback_manager import FeedbackManager
import base64, cv2, numpy as np, json
import time, uuid
from app.core.redis import redis_client
from app.config.feedback_criteria import PresentationType
from app.services.whisper_service import whisper_service
import tempfile
import wave
import os

router = APIRouter(tags=["Realtime Analysis"])

@router.websocket("/realtime")
async def realtime_socket(ws: WebSocket):
    """
    실시간 분석 WebSocket
    - frame + audio(base64) 형태로 받음
    - 분석 결과 + 피드백 실시간 전송
    """
    silence_start_time = None
    SILENCE_THRESHOLD = 200
    SILENCE_LIMIT = 3.0  # 3초 이상 정적이면 피드백

    audio_buffer = []
    audio_chunk_duration = 1.0   # 프론트가 1초 chunk 보낸다고 가정
    last_stt_time = time.time()
    STT_INTERVAL = 5.0           # 5초마다 STT 실행


    await ws.accept()
    print("[WebSocket] 연결 시작")
    presentation_type = ws.query_params.get("type", "small")
    feedback_manager = FeedbackManager(presentation_type=presentation_type) # 또는 SMALL / ONLINE_SMALL -> 스프링 연결 필요
    presentation_id = uuid.uuid4().hex
    presentation_start_time = time.time()
    await ws.send_text(json.dumps({
        "type": "session_start",
        "presentationId": presentation_id
    }, ensure_ascii=False))

    try:
        while True:
            # 클라이언트로부터 랜드마크 JSON 수신
            # { "face": {...}, "pose": {...}, "audio": "base64_audio_chunk", "timestamp": 1710000000 }
            data = await ws.receive_json()
            audio_base64 = data.get("audio")
            if audio_base64:
                audio_np = analyzer.decode_audio(audio_base64)
                audio_buffer.append(audio_np)
                # 오디오 청크 받을 때 볼륨 계산
                volume = np.abs(audio_np).mean()

                if volume < SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                else:
                    silence_start_time = None

        
            raw_result = analyzer.analyze_realtime_landmarks(data)
            
            #정적이 일정시간 넘으면 feedback 생성
            if silence_start_time:
                silence_duration = time.time() - silence_start_time

                if silence_duration > SILENCE_LIMIT:
                    raw_result["speech"] = {
                        "text": "",
                        "wpm": 0,
                        "fillers_freq": 0,
                        "silence": True
                    }
            current_time = time.time()
            speech_result = None
            
            if current_time - last_stt_time > STT_INTERVAL and len(audio_buffer) >= 3:
                full_audio = np.concatenate(audio_buffer)
                # 임시 wav 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    wav_path = tmp.name
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(full_audio.tobytes())

                # WhisperService 호출
                try:
                    whisper_service.transcribe_async(wav_path)
                    whisper_res = whisper_service.get_result()

                    if whisper_res["status"] == "success":
                        speech_result = whisper_res["data"]
                except Exception as e:
                    print("whisper error: ",e)
                
                finally:
                    os.remove(wav_path)
                # 버퍼 초기화
                audio_buffer.clear()
                last_stt_time = current_time
            
            if speech_result:
                raw_result["speech"] = speech_result
            manager_feedback = feedback_manager.update(raw_result)
           
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
