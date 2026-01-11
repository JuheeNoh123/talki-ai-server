# app/services/whisper_service.py
from test_record_multiprocess import analyze_parallel, WhisperService

# Whisper 모델 1회만 로드
whisper_service = WhisperService()
whisper_service.start()

def analyze_record_video(video_path: str):
    """
    녹화 영상 전체 분석
    - WhisperService를 병렬 분석기에 주입
    - Whisper는 이미 로드되어 있으므로 중복 로딩 없음
    """
    print(f"[Analyze Service] 녹화 영상 분석 요청: {video_path}")
    result = analyze_parallel(video_path, whisper_service)
    return result
