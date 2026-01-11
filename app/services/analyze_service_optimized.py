# app/services/analyze_service_optimized.py
import time
import json
from .whisper_service import WhisperService
from test_record_multiprocess import analyze_parallel
# Whisper 모델 1회 로드
whisper_service = WhisperService()
whisper_service.start()

def analyze_record_video(video_path: str):
    """녹화 영상 전체 분석"""
    print(f"[Analyze Service] 녹화 영상 분석 요청: {video_path}")
    return analyze_parallel(video_path, whisper_service)

