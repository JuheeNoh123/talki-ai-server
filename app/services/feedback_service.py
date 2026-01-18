# app/services/feedback_service.py
import numpy as np
from app.config.feedback_criteria import (
    PresentationType,
    FEEDBACK_CRITERIA,
)


def clamp(score: float) -> int:
    return int(max(0, min(100, score)))


def calc_gaze_score(front_ratio: float, criteria: dict) -> int:
    std = criteria["gaze_front_ratio"]
    if front_ratio >= std:
        return 100
    diff = std - front_ratio
    return clamp(100 - (diff / std) * 100)


def calc_wpm_score(wpm: float, criteria: dict) -> int:
    min_wpm = criteria["wpm_min"]
    max_wpm = criteria["wpm_max"]

    if wpm == 0:
        return 0
    if min_wpm <= wpm <= max_wpm:
        return 100

    if wpm < min_wpm:
        diff = (min_wpm - wpm) / min_wpm
    else:
        diff = (wpm - max_wpm) / max_wpm

    return clamp(100 - diff * 100)


def calc_filler_score(fillers: int, criteria: dict) -> int:
    allowed = criteria["fillers_per_min"]
    if fillers <= allowed:
        return 100
    return clamp(100 - (fillers - allowed) * 15)


def calc_pose_score(avg_speed: float, criteria: dict) -> int:
    min_s = criteria["pose_min"]
    max_s = criteria["pose_max"]

    if min_s <= avg_speed <= max_s:
        return 100

    if avg_speed > max_s:
        diff = (avg_speed - max_s) / max_s
    else:
        diff = (min_s - avg_speed) / min_s

    return clamp(100 - diff * 100)

def generate_feedback(analysis_result: dict, presentation_type: str = PresentationType.SMALL):
    criteria = FEEDBACK_CRITERIA[presentation_type]
    feedback = []

    # --- 시선 피드백 ---
    gaze = analysis_result.get("eyes", {})
    horiz_counts = gaze.get("horiz_counts", {})
    vert_mode = gaze.get("vert_mode")
    samples = gaze.get("samples", 0)

    front_ratio = (
        horiz_counts.get("center", 0) / samples
        if samples > 0 else 0.0
    )

    if front_ratio < criteria["gaze_front_ratio"]:
        feedback.append("정면을 바라보는 시간이 부족합니다. 청중을 더 자주 바라봐주세요.")
    elif vert_mode in ("up", "down"):
        feedback.append("시선이 다소 불안정합니다. 정면을 유지해 주세요.")
    else:
        feedback.append("시선이 안정적입니다. 좋아요!")


    # --- 자세/동작 피드백 ---
    avg_speed = analysis_result.get("handArmMovementAvg", 0.0)

    if avg_speed > criteria["pose_max"]:
        feedback.append("몸을 비교적 많이 움직이는 편입니다. 조금 더 차분한 자세를 취해보세요.")
    elif avg_speed < criteria["pose_min"]:
        feedback.append("자세가 다소 경직되어 있습니다. 자연스러운 제스처를 사용해 보세요.")
    else:
        feedback.append("자세가 안정적으로 유지되었습니다.")

    # --- 음성 피드백 ---
    wpm = analysis_result.get("WPM", 0)
    speech = analysis_result.get("speech", {})
    fillers = speech.get("fillers_freq", 0)

    if wpm == 0:
        feedback.append("음성 입력이 없습니다. 마이크 상태를 확인해주세요.")
    elif wpm < criteria["wpm_min"]:
        feedback.append("말이 조금 느립니다. 자신감 있게 말씀해 보세요.")
    elif wpm > criteria["wpm_max"]:
        feedback.append("말이 너무 빠릅니다. 조금 천천히 말씀해 보세요.")
    else:
        feedback.append("적절한 말 속도입니다. 좋습니다!")

    if fillers > criteria["fillers_per_min"]:
        feedback.append(
            "습관적인 추임새(음, 어)가 다소 잦습니다. 의식적으로 줄여보세요."
        )
    # --- 종합 요약 ---
    gaze_score = calc_gaze_score(front_ratio, criteria)
    speech_score = calc_wpm_score(wpm, criteria)
    filler_score = calc_filler_score(fillers, criteria)
    pose_score = calc_pose_score(avg_speed, criteria)

    total_score = round(
        gaze_score * 0.30 +
        speech_score * 0.25 +
        filler_score * 0.15 +
        pose_score * 0.30
    )
    # 상세 수치 반환
    return {
        "score": total_score,
        "score_detail": {
            "gaze": gaze_score,
            "speech_speed": speech_score,
            "fillers": filler_score,
            "pose": pose_score,
        },
        "feedback_text": " / ".join(feedback),
        "metrics": {
            "gaze_front_ratio": round(front_ratio, 2),
            "pose_avg_speed": round(avg_speed, 4),
            "speech_wpm": round(wpm, 1),
            "speech_fillers": fillers,
        }
    }


