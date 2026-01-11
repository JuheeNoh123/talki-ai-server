# app/services/analyze_service_landmarks.py
from app.utils.analysis_utils import gaze_from_landmarks

def analyze_realtime_landmarks(data: dict):
    """
    클라이언트에서 보낸 랜드마크 데이터를 분석합니다.
    data = { "face": {...}, "pose": {...}, ... }
    """
    feedback = {}

    # 1. 시선 분석 (Shared Logic)
    face_lms = data.get("face")
    if face_lms:
        # analysis_utils supports dict input specifically for this case
        feedback["gaze"] = gaze_from_landmarks(face_lms)

    # 2. 자세 분석 (데이터 패스스루)
    pose_lms = data.get("pose")
    if pose_lms:
        feedback["pose_detected"] = True
        # FeedbackManager expects format compat with utils (optional, but good to be compliant)
        # However, FeedbackManager itself uses movement_speed (maybe we should use that there too?)
        # For now, we just pass raw data.
        
        # NOTE: data["pose"] keys are "13", "14" strings.
        # FeedbackManager might need ints or handle strings.
        # Our updated movement_speed in utils handles strings.
        # But let's check FeedbackManager implementation again to be sure.
        feedback["pose_landmarks"] = pose_lms
    else:
        feedback["pose_detected"] = False

    return feedback
