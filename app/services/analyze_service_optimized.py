# app/services/analyze_service_optimized.py
from .whisper_service import whisper_service
from test_record_multiprocess import analyze_parallel

def analyze_record_video(video_path: str):
    """녹화 영상 전체 분석 — 공유 singleton whisper_service 사용"""
    print(f"[Analyze Service] 녹화 영상 분석 요청: {video_path}")
    return analyze_parallel(video_path, whisper_service)

