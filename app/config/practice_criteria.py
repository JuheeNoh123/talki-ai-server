class PracticeSubStep:
    SCRIPT = "SCRIPT"
    GAZE = "GAZE"
    IMPROMPTU = "IMPROMPTU"
    KEYWORD = "KEYWORD"
    POINT = "POINT"


AUDIO_SUBSTEPS = {
    PracticeSubStep.SCRIPT,
    PracticeSubStep.IMPROMPTU,
    PracticeSubStep.KEYWORD,
    PracticeSubStep.POINT,
}

# 스크립트 읽기 연습에서 사용할 사전 정의 스크립트 풀 (LLM 미사용)
SCRIPT_POOL = [
    "안녕하세요. 오늘은 짧지만 중요한 이야기를 해보려고 합니다. 우리는 하루에도 정말 많은 말을 하지만, "
    "그 말이 상대에게 어떻게 전달되는지까지는 신경 쓰지는 못합니다. 같은 내용이라도 말하는 속도, 목소리의 크기, "
    "잠깐의 멈춤에 따라 분위기는 완전히 달라질 수 있습니다. 그래서 좋은 전달은 단순히 정확하게 읽는 것이 아니라, "
    "듣는 사람이 편하게 이해할 수 있도록 말하는 것이라고 생각합니다. 천천히 또렷하게 말하면 자신감 있어 보이고, "
    "적절한 호흡과 자연스러운 억양을 더하면 내용도 훨씬 잘 전달됩니다. 결국 좋은 말하기란 어려운 기술이 아니라, "
    "상대를 배려하는 작은 태도에서 시작된다고 생각합니다.",
]

# subStep별 WPM 판정 기준값 (발표용 FEEDBACK_CRITERIA와는 별도 — 낭독/즉흥은 기준 범위가 다름)
PRACTICE_CRITERIA = {
    PracticeSubStep.SCRIPT: {"wpm_min": 120, "wpm_max": 160},
    PracticeSubStep.IMPROMPTU: {"wpm_min": 100, "wpm_max": 170},
    PracticeSubStep.KEYWORD: {"wpm_min": 100, "wpm_max": 170},
    PracticeSubStep.POINT: {"wpm_min": 100, "wpm_max": 170},
}

SPEAK_SECONDS = {
    PracticeSubStep.SCRIPT: 60,
    PracticeSubStep.IMPROMPTU: 30,
    PracticeSubStep.KEYWORD: 30,
    PracticeSubStep.POINT: 30,
}

PREP_SECONDS = {
    PracticeSubStep.IMPROMPTU: 10,
    PracticeSubStep.KEYWORD: 10,
}

GAZE_TARGET_DURATION_SEC = 30

# 오디오 스트리밍 버퍼링 주기 (실전 탭 realtime_router.py의 STT_INTERVAL과 동일 패턴)
STT_INTERVAL_SEC = 5.0
MIN_CHUNKS_BEFORE_STT = 3

# 준비(prep) 구간 동안 클라이언트가 프레임을 보내지 않을 수 있어 여유를 둔 수신 타임아웃
RECEIVE_TIMEOUT_SEC = 15.0
