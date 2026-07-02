# app/routers/websocket_router.py
from fastapi import APIRouter, WebSocket
from app.services import analyze_service_landmarks as analyzer
from app.services.feedback_manager import FeedbackManager
from app.services.surprise_question_manager import SurpriseQuestionManager
from app.services import surprise_question_service
from starlette.websockets import WebSocketDisconnect

import asyncio
import re
import base64, cv2, numpy as np, json
import time, uuid
from datetime import datetime, timezone

from app.core.redis import redis_client
from app.core.scorer import scorer
from app.config.feedback_criteria import PresentationType
from app.services.whisper_service import whisper_service

import tempfile
import wave
import os


# ── Redis 저장 헬퍼 ───────────────────────────────────────────────────────────

async def save_segment(presentation_id, seg_type, start, end):
    event = {
        "type": seg_type,
        "start": round(start, 1),
        "end": round(end, 1),
        "duration": round(end - start, 1),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    }
    try:
        await redis_client.rpush(
            f"presentation:{presentation_id}:segments",
            json.dumps(event, ensure_ascii=False)
        )
        await redis_client.expire(f"presentation:{presentation_id}:segments", 60 * 60)
    except Exception as e:
        print(f"[Redis] save_segment 실패 (무시): {e}")


async def save_question_pending(presentation_id: str, question_id: str,
                                 question: str, asked_at_seconds: float, context_text: str):
    """질문 전송 시 타임스탬프와 함께 Redis에 저장 (배치 분석에서 답변 평가에 사용)."""
    record = {
        "question_id":      question_id,
        "question":         question,
        "asked_at_seconds": round(asked_at_seconds, 1),
        "context_text":     context_text,
    }
    try:
        await redis_client.rpush(
            f"presentation:{presentation_id}:surprise_questions_pending",
            json.dumps(record, ensure_ascii=False)
        )
        await redis_client.expire(
            f"presentation:{presentation_id}:surprise_questions_pending", 60 * 60
        )
    except Exception as e:
        print(f"[Redis] save_question_pending 실패 (무시): {e}")


# ── WAV 파일 생성 헬퍼 ────────────────────────────────────────────────────────

def _write_wav(audio_frames: list) -> str:
    full_audio = np.concatenate(audio_frames)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(full_audio.tobytes())
    return wav_path


# ── 라우터 ────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["Realtime Analysis"])


@router.websocket("/realtime")
async def realtime_socket(ws: WebSocket):

    # ── 기존 상태 변수 ────────────────────────────────────────────────────────
    active_segments = {
        "speech_fast": None,
        "speech_slow": None,
        "pose_rigid": None,
        "pose_unstable": None,
        "gaze_unstable": None,
        "silence": None,
    }

    SILENCE_THRESHOLD    = 200
    SILENCE_LIMIT        = 3.0
    audio_buffer  = []
    last_stt_time = time.time()
    STT_INTERVAL         = 5.0
    RECEIVE_TIMEOUT      = 3.0

    await ws.accept()
    print("[WebSocket] 연결 시작")

    presentation_type       = ws.query_params.get("type", "small")
    feedback_manager        = FeedbackManager(presentation_type=presentation_type)
    presentation_id         = uuid.uuid4().hex
    presentation_start_time = time.time()

    await ws.send_text(json.dumps({
        "type": "session_start",
        "presentationId": presentation_id
    }, ensure_ascii=False))

    # ── 돌발 질문 상태 변수 ───────────────────────────────────────────────────
    sq_manager = SurpriseQuestionManager()
    sq_manager.start(presentation_start_time)

    try:
        while True:
            current_time = time.time()
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                print("[WebSocket] 수신 타임아웃 — 클라이언트 연결 종료로 판단")
                break

            # ── 1. 랜드마크 기반 분석 (시선 + 자세) ← 기존 그대로 ────────────
            raw_result = analyzer.analyze_realtime_landmarks(data)

            # ── 2. 오디오 처리 ← 기존 그대로 ────────────────────────────────
            audio_base64 = data.get("audio")
            if audio_base64:
                audio_np = analyzer.decode_audio(audio_base64)
                volume   = np.abs(audio_np).mean()
                audio_buffer.append(audio_np)

                if volume < SILENCE_THRESHOLD:
                    if active_segments["silence"] is None:
                        active_segments["silence"] = current_time
                    if "speech" not in raw_result:
                        raw_result["speech"] = {}
                    raw_result["speech"]["silence"] = True
                else:
                    if active_segments["silence"] is not None:
                        silence_duration = current_time - active_segments["silence"]
                        if silence_duration > SILENCE_LIMIT:
                            await save_segment(
                                presentation_id, "silence",
                                active_segments["silence"] - presentation_start_time,
                                current_time - presentation_start_time,
                            )
                    active_segments["silence"] = None

            # ── 3. Whisper STT ← 기존 그대로 ─────────────────────────────────
            speech_result = None
            if current_time - last_stt_time > STT_INTERVAL and len(audio_buffer) >= 3:
                wav_path = _write_wav(audio_buffer)
                try:
                    loop = asyncio.get_event_loop()
                    whisper_res = await loop.run_in_executor(
                        None, whisper_service.transcribe, wav_path
                    )
                    if whisper_res["status"] == "success":
                        speech_result = whisper_res["data"]
                except Exception as e:
                    print("whisper error:", e)
                finally:
                    os.remove(wav_path)

                audio_buffer.clear()
                last_stt_time = current_time

            # ── 4. 말 속도 구간 감지 ← 기존 그대로 ──────────────────────────
            if speech_result:
                if "speech" not in raw_result:
                    raw_result["speech"] = {}
                raw_result["speech"].update(speech_result)

                wpm = speech_result.get("wpm", 0)

                if wpm > feedback_manager.criteria["wpm_max"]:
                    if active_segments["speech_fast"] is None:
                        active_segments["speech_fast"] = current_time - presentation_start_time
                    if active_segments["speech_slow"] is not None:
                        await save_segment(presentation_id, "speech_slow",
                                           active_segments["speech_slow"],
                                           current_time - presentation_start_time)
                        active_segments["speech_slow"] = None

                elif wpm < feedback_manager.criteria["wpm_min"] and wpm > 0:
                    if active_segments["speech_slow"] is None:
                        active_segments["speech_slow"] = current_time - presentation_start_time
                    if active_segments["speech_fast"] is not None:
                        await save_segment(presentation_id, "speech_fast",
                                           active_segments["speech_fast"],
                                           current_time - presentation_start_time)
                        active_segments["speech_fast"] = None

                else:
                    if active_segments["speech_fast"] is not None:
                        await save_segment(presentation_id, "speech_fast",
                                           active_segments["speech_fast"],
                                           current_time - presentation_start_time)
                        active_segments["speech_fast"] = None
                    if active_segments["speech_slow"] is not None:
                        await save_segment(presentation_id, "speech_slow",
                                           active_segments["speech_slow"],
                                           current_time - presentation_start_time)
                        active_segments["speech_slow"] = None

                # [신규] STT 완료 후 돌발 질문 트리거 체크
                sq_manager.add_stt(speech_result.get("text", ""))
                print(f"[SurpriseQ] STT 텍스트: '{speech_result.get('text', '')}' / 누적: {sq_manager.accumulated_words}단어")  # 추가
                if sq_manager.should_trigger(current_time):
                    elapsed = round(current_time - presentation_start_time, 1)
                    print(f"[SurpriseQ] 트리거 발동 — 누적 단어: {sq_manager.accumulated_words}개, 경과: {elapsed}초")
                    sq_manager.start_generating(current_time)
                    asyncio.create_task(_send_question(
                        ws, sq_manager,
                        presentation_id, presentation_start_time,
                    ))

            # ── 5. 시선/자세 피드백 업데이트 ← 기존 그대로 ──────────────────
            manager_feedback = feedback_manager.update(raw_result)

            # ── 6. 자세 경직/산만 구간 감지 ← 기존 그대로 ───────────────────
            pose_landmarks = raw_result.get("pose_landmarks")
            if pose_landmarks:
                avg_speed = np.mean(feedback_manager.movement_speeds) if feedback_manager.movement_speeds else 0

                if avg_speed < feedback_manager.criteria["pose_min"]:
                    if active_segments["pose_rigid"] is None:
                        active_segments["pose_rigid"] = current_time
                else:
                    if active_segments["pose_rigid"] is not None:
                        await save_segment(presentation_id, "pose_rigid",
                                           active_segments["pose_rigid"] - presentation_start_time,
                                           current_time - presentation_start_time)
                        active_segments["pose_rigid"] = None

                if avg_speed > feedback_manager.criteria["pose_max"]:
                    if active_segments["pose_unstable"] is None:
                        active_segments["pose_unstable"] = current_time
                else:
                    if active_segments["pose_unstable"] is not None:
                        await save_segment(presentation_id, "pose_unstable",
                                           active_segments["pose_unstable"] - presentation_start_time,
                                           current_time - presentation_start_time)
                        active_segments["pose_unstable"] = None

            # ── 7. 시선 불안정 구간 감지 ← 기존 그대로 ──────────────────────
            if raw_result.get("gaze_unstable"):
                if active_segments["gaze_unstable"] is None:
                    active_segments["gaze_unstable"] = current_time
            else:
                if active_segments["gaze_unstable"] is not None:
                    await save_segment(presentation_id, "gaze_unstable",
                                       active_segments["gaze_unstable"] - presentation_start_time,
                                       current_time - presentation_start_time)
                    active_segments["gaze_unstable"] = None

            # ── 8. 실시간 피드백 전송 ← 기존 그대로 ──────────────────────────
            await ws.send_text(json.dumps({
                "type": "feedback",
                "raw_result": raw_result,
                "data": manager_feedback,
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        print("[WebSocket] 클라이언트 정상 종료")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WebSocket] 비정상 종료: {e}")

    # ── 9. 발표 종료 시 열려있는 segment 정리 ← 기존 그대로 ─────────────────
    finally:
        now         = time.time()
        elapsed_now = now - presentation_start_time
        already_elapsed = {"speech_fast", "speech_slow"}

        for seg_type, start_time in active_segments.items():
            if start_time is not None:
                start_elapsed = start_time if seg_type in already_elapsed else start_time - presentation_start_time
                await save_segment(presentation_id, seg_type, start_elapsed, elapsed_now)

        try:
            if ws.application_state.name not in ('DISCONNECTED',) and ws.client_state.name != 'DISCONNECTED':
                await ws.close()
        except RuntimeError:
            pass


# ── STT 오인식 문장 필터 ──────────────────────────────────────────────────────

async def _filter_context(context: str, quality_threshold: float = 0.3) -> str:
    """quality_score 기반으로 오인식/헛소리 문장을 제거한 컨텍스트 반환."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=다)\s+", context) if s.strip()]
    if not sentences:
        return context
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, scorer._batch_quality_scores, sentences)
    filtered = [s for s, q in zip(sentences, scores) if q >= quality_threshold]
    print(f"[SurpriseQ] 컨텍스트 필터: {len(sentences)}문장 → {len(filtered)}문장 (threshold={quality_threshold})")
    return " ".join(filtered) if filtered else context


# ── 돌발 질문 전송 (백그라운드 태스크) ───────────────────────────────────────

async def _send_question(
    ws, sq_manager: SurpriseQuestionManager,
    presentation_id: str, presentation_start_time: float,
):
    raw_context = sq_manager.get_context()
    context = await _filter_context(raw_context)
    if not context.strip():
        context = raw_context
    try:
        print(f"[SurpriseQ] GPT 질문 생성 중... (컨텍스트 {len(context.split())}단어)")
        q_result = await surprise_question_service.generate_question(context, sq_manager.asked_questions)
        if not q_result:
            print("[SurpriseQ] 질문 생성 실패 — IDLE 복귀")
            sq_manager.reset_to_idle()
            return

        question_id = uuid.uuid4().hex
        asked_at_seconds = time.time() - presentation_start_time

        # Redis에 질문 + 타임스탬프 저장 (배치 분석에서 답변 위치 특정에 사용)
        await save_question_pending(
            presentation_id, question_id,
            q_result["question"], asked_at_seconds, context,
        )

        # 클라이언트에 질문 전송
        await ws.send_text(json.dumps({
            "type":        "surprise_question",
            "question_id": question_id,
            "question":    q_result["question"],
            "time_limit":  30,
        }, ensure_ascii=False))

        print(f"[SurpriseQ] ✅ 질문 전송 완료")
        print(f"[SurpriseQ]    question_id      : {question_id}")
        print(f"[SurpriseQ]    question         : {q_result['question']}")
        print(f"[SurpriseQ]    asked_at_seconds : {round(asked_at_seconds, 1)}초")

        # 즉시 IDLE 복귀 → 다음 질문 준비
        sq_manager.set_triggered(q_result["question"])

    except Exception as e:
        print(f"[SurpriseQ] 질문 생성 태스크 실패: {e}")
        sq_manager.reset_to_idle()
