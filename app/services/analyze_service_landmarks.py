# app/services/analyze_service_landmarks.py
from app.utils.analysis_utils import gaze_from_landmarks
import base64, cv2, numpy as np, json

def analyze_realtime_landmarks(data: dict):
    """
    클라이언트에서 보낸 랜드마크 데이터를 분석합니다.
    data = { "face": {...}, "pose": {...}, ... }
    """
    feedback = {}

    # 1. 시선 분석
    face_lms = data.get("face")
    if face_lms:
        gaze = gaze_from_landmarks(face_lms)
        # 클라이언트(JS MediaPipe) Y축은 Python MediaPipe와 반대
        # (JS: y=0이 아래, Python: y=0이 위) → 수직 방향 반전
        if gaze["vert"] == "up":
            gaze["vert"] = "down"
        elif gaze["vert"] == "down":
            gaze["vert"] = "up"
        feedback["gaze"] = gaze

    # 2. 자세 분석 (데이터 패스스루)
    pose_lms = data.get("pose")
    if pose_lms:
        feedback["pose_detected"] = True        
        feedback["pose_landmarks"] = pose_lms
    else:
        feedback["pose_detected"] = False

    return feedback

def decode_audio(audio_base64):
    audio_bytes = base64.b64decode(audio_base64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_np