# app/routers/practice_router.py
# 연습 탭(CBT 6단계) 4단계 "연습 진행" — Spring과의 단일 WebSocket 계약.
# docs/practice-tab-fastapi-spec.md §2~§6 참고.
import asyncio
import json
import os
import random
import tempfile
import time
import wave

import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.config.practice_criteria import (
    AUDIO_SUBSTEPS,
    GAZE_TARGET_DURATION_SEC,
    MIN_CHUNKS_BEFORE_STT,
    PRACTICE_CRITERIA,
    PREP_SECONDS,
    PracticeSubStep,
    RECEIVE_TIMEOUT_SEC,
    SCRIPT_POOL,
    SPEAK_SECONDS,
    STT_INTERVAL_SEC,
)
from app.services import practice_llm_service
from app.services.analyze_service_landmarks import analyze_realtime_landmarks, decode_audio
from app.services.practice_analysis_service import (
    GazeFixationTracker,
    articulation_label,
    build_gaze_feedback_text,
    build_script_feedback,
    build_script_final_feedback_text,
    calc_gaze_score,
    calc_impromptu_score,
    calc_keyword_score,
    calc_point_score,
    calc_script_score,
    keyword_usage,
)
from app.services.whisper_service import whisper_service

router = APIRouter(tags=["Practice"])

_EMPTY_STATS = {
    "text": "", "wpm": 0.0, "fillers_count": 0, "fillers_freq": 0.0,
    "duration": 0.0, "avg_word_probability": 0.0,
}


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


async def _build_session_start(sub_step: str) -> dict:
    payload = {"type": "session_start", "subStep": sub_step}

    if sub_step == PracticeSubStep.SCRIPT:
        payload["script_text"] = random.choice(SCRIPT_POOL)
        payload["speak_seconds"] = SPEAK_SECONDS[sub_step]
        criteria = PRACTICE_CRITERIA[sub_step]
        payload["reference_range"] = {"wpm_min": criteria["wpm_min"], "wpm_max": criteria["wpm_max"]}

    elif sub_step == PracticeSubStep.GAZE:
        payload["target_duration_sec"] = GAZE_TARGET_DURATION_SEC

    elif sub_step == PracticeSubStep.IMPROMPTU:
        result = await practice_llm_service.generate_impromptu_topic()
        payload["topic"] = result["question"]
        payload["prep_seconds"] = PREP_SECONDS[sub_step]
        payload["speak_seconds"] = SPEAK_SECONDS[sub_step]

    elif sub_step == PracticeSubStep.KEYWORD:
        result = await practice_llm_service.generate_keywords()
        payload["keywords"] = result["keywords"]
        payload["prep_seconds"] = PREP_SECONDS[sub_step]
        payload["speak_seconds"] = SPEAK_SECONDS[sub_step]

    elif sub_step == PracticeSubStep.POINT:
        result = await practice_llm_service.generate_gist_passage()
        payload["passage"] = result["source_text"]
        payload["reference_keywords"] = result["reference_keywords"]
        payload["speak_seconds"] = SPEAK_SECONDS[sub_step]

    return payload


async def _build_result(
    sub_step: str,
    session_start_payload: dict,
    gaze_tracker: GazeFixationTracker | None,
    full_audio_buffer: list,
    latest_stats: dict | None,
    loop,
) -> dict:
    if sub_step == PracticeSubStep.GAZE:
        stats = gaze_tracker.finalize(time.time()) if gaze_tracker else {
            "gaze_hold_ratio": 0.0, "avg_hold_duration_sec": 0.0, "gaze_break_count": 0,
        }
        score = calc_gaze_score(stats["gaze_hold_ratio"], stats["gaze_break_count"])
        return {
            "type": "result",
            "subStep": sub_step,
            "score": score,
            "feedback_text": build_gaze_feedback_text(stats["gaze_hold_ratio"]),
            "raw_result": stats,
        }

    # 음성 기반 subStep: 스트리밍 중 모은 오디오 전체로 최종 STT 재분석
    final_stats = None
    if full_audio_buffer:
        wav_path = _write_wav(full_audio_buffer)
        try:
            whisper_res = await loop.run_in_executor(None, whisper_service.transcribe, wav_path)
            if whisper_res["status"] == "success":
                final_stats = whisper_res["data"]
        except Exception as e:
            print(f"[Practice] 최종 STT 실패: {e}")
        finally:
            os.remove(wav_path)

    final_stats = final_stats or latest_stats or _EMPTY_STATS

    wpm = final_stats.get("wpm", 0.0)
    fillers_count = final_stats.get("fillers_count", 0)
    fillers_freq = final_stats.get("fillers_freq", 0.0)
    duration_sec = final_stats.get("duration", 0.0)
    text = final_stats.get("text", "") or ""
    avg_word_probability = final_stats.get("avg_word_probability", 0.0)
    articulation = articulation_label(avg_word_probability)

    raw_result = {
        "wpm": wpm,
        "fillers_count": fillers_count,
        "fillers_freq": fillers_freq,
        "duration_sec": duration_sec,
        "articulation_label": articulation,
        "text": text,
    }

    criteria = PRACTICE_CRITERIA.get(sub_step, PRACTICE_CRITERIA[PracticeSubStep.SCRIPT])

    if sub_step == PracticeSubStep.SCRIPT:
        score = calc_script_score(wpm, fillers_freq, avg_word_probability, criteria["wpm_min"], criteria["wpm_max"])
        feedback_text = build_script_final_feedback_text(wpm, articulation, criteria["wpm_min"], criteria["wpm_max"])

    elif sub_step == PracticeSubStep.IMPROMPTU:
        topic = session_start_payload.get("topic", "")
        speak_seconds = session_start_payload.get("speak_seconds", 30)
        spoken_ratio = min(1.0, duration_sec / speak_seconds) if speak_seconds else 0.0
        if text.strip():
            struct_eval = await practice_llm_service.evaluate_structure_completeness(topic, text)
            structure_label = struct_eval["structure_completeness"]
        else:
            structure_label = "미흡"
        score = calc_impromptu_score(spoken_ratio, fillers_freq, structure_label)
        raw_result["spoken_duration_sec"] = round(duration_sec, 1)
        raw_result["structure_completeness"] = structure_label
        feedback_text = (
            f"주어진 시간을 {'거의 다' if spoken_ratio >= 0.8 else '일부만'} 활용했고, "
            f"구성은 {structure_label} 수준입니다."
        )

    elif sub_step == PracticeSubStep.KEYWORD:
        keywords = session_start_payload.get("keywords", [])
        usage = keyword_usage(keywords, text)
        used_count = sum(1 for u in usage if u["used"])
        if text.strip() and keywords:
            naturalness_eval = await practice_llm_service.evaluate_keyword_connection(keywords, text)
            naturalness_label = naturalness_eval["naturalness_label"]
        else:
            naturalness_label = "미흡"
        score = calc_keyword_score(used_count, len(keywords), naturalness_label)
        raw_result["keyword_usage"] = usage
        raw_result["keyword_coverage"] = f"{used_count}/{len(keywords)}"
        raw_result["connection_naturalness"] = naturalness_label
        feedback_text = f"키워드 {used_count}/{len(keywords)}개를 사용했고, 연결은 {naturalness_label} 수준입니다."

    elif sub_step == PracticeSubStep.POINT:
        reference_keywords = session_start_payload.get("reference_keywords", [])
        source_text = session_start_payload.get("passage", "")
        usage = keyword_usage(reference_keywords, text)
        used_count = sum(1 for u in usage if u["used"])
        if text.strip() and reference_keywords:
            gist_eval = await practice_llm_service.evaluate_gist_accuracy(reference_keywords, source_text, text)
            gist_label = gist_eval["gist_accuracy_label"]
            missing_points = gist_eval.get("missing_points", [])
        else:
            gist_label = "미흡"
            missing_points = reference_keywords
        score = calc_point_score(used_count, len(reference_keywords), gist_label)
        raw_result["reference_keyword_coverage"] = f"{used_count}/{len(reference_keywords)}"
        raw_result["gist_accuracy_label"] = gist_label
        raw_result["missing_points"] = missing_points
        feedback_text = f"핵심 파악 정확도는 {gist_label} 수준입니다."

    else:
        score, feedback_text = 0, ""

    return {
        "type": "result",
        "subStep": sub_step,
        "score": score,
        "feedback_text": feedback_text,
        "raw_result": raw_result,
    }


@router.websocket("/practice/realtime")
async def practice_realtime(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("sessionId", "unknown")
    sub_step = ws.query_params.get("subStep", PracticeSubStep.SCRIPT)
    print(f"[Practice] 연결 시작 sessionId={session_id} subStep={sub_step}", flush=True)

    session_start_payload: dict = {}
    duration_limit = None
    speech_start_time = None
    audio_buffer: list = []
    full_audio_buffer: list = []
    last_stt_time = time.time()
    latest_stats: dict | None = None
    gaze_tracker = GazeFixationTracker() if sub_step == PracticeSubStep.GAZE else None
    gaze_start_time = None
    loop = asyncio.get_event_loop()

    try:
        session_start_payload = await _build_session_start(sub_step)
        await ws.send_text(json.dumps(session_start_payload, ensure_ascii=False))
        print(f"[Practice] session_start 전송 완료: {session_start_payload}", flush=True)

        duration_limit = session_start_payload.get("speak_seconds") or session_start_payload.get("target_duration_sec")

        while True:
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=RECEIVE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                print("[Practice] 수신 타임아웃 — 세션 종료로 판단")
                break

            now = time.time()

            if sub_step == PracticeSubStep.GAZE:
                face = data.get("face")
                if face:
                    if gaze_start_time is None:
                        gaze_start_time = now
                    raw = analyze_realtime_landmarks({"face": face})
                    gaze = raw.get("gaze")
                    if gaze and gaze_tracker:
                        gaze_tracker.update(gaze, now)
                if gaze_start_time and duration_limit and now - gaze_start_time >= duration_limit:
                    break

            elif sub_step in AUDIO_SUBSTEPS:
                audio_b64 = data.get("audio")
                if audio_b64:
                    if speech_start_time is None:
                        speech_start_time = now
                    audio_np = decode_audio(audio_b64)
                    audio_buffer.append(audio_np)
                    full_audio_buffer.append(audio_np)

                    if now - last_stt_time > STT_INTERVAL_SEC and len(audio_buffer) >= MIN_CHUNKS_BEFORE_STT:
                        wav_path = _write_wav(audio_buffer)
                        try:
                            whisper_res = await loop.run_in_executor(None, whisper_service.transcribe, wav_path)
                            if whisper_res["status"] == "success":
                                latest_stats = whisper_res["data"]
                                criteria = PRACTICE_CRITERIA.get(sub_step, PRACTICE_CRITERIA[PracticeSubStep.SCRIPT])
                                feedback_msgs = build_script_feedback(
                                    latest_stats.get("wpm", 0.0),
                                    latest_stats.get("fillers_freq", 0.0),
                                    criteria["wpm_min"],
                                    criteria["wpm_max"],
                                )
                                if feedback_msgs:
                                    await ws.send_text(json.dumps(
                                        {"type": "feedback", "subStep": sub_step, "data": feedback_msgs},
                                        ensure_ascii=False,
                                    ))
                        except Exception as e:
                            print(f"[Practice] whisper 처리 실패: {e}")
                        finally:
                            os.remove(wav_path)
                        audio_buffer.clear()
                        last_stt_time = now

                if speech_start_time and duration_limit and now - speech_start_time >= duration_limit:
                    break

    except WebSocketDisconnect:
        print("[Practice] 클라이언트 정상 종료", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Practice] 비정상 종료: {e}", flush=True)

    print(f"[Practice] 결과 계산 시작 sub_step={sub_step}", flush=True)
    try:
        result_payload = await _build_result(
            sub_step, session_start_payload, gaze_tracker, full_audio_buffer, latest_stats, loop,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Practice] 결과 계산 실패: {e}", flush=True)
        result_payload = {"type": "result", "subStep": sub_step, "score": 0, "feedback_text": "", "raw_result": {}}

    try:
        await ws.send_text(json.dumps(result_payload, ensure_ascii=False))
        print(f"[Practice] result 전송 완료: {result_payload}", flush=True)
    except Exception as e:
        print(f"[Practice] 결과 전송 실패: {e}", flush=True)

    try:
        if ws.application_state.name != "DISCONNECTED" and ws.client_state.name != "DISCONNECTED":
            await ws.close()
    except RuntimeError:
        pass
