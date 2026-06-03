import uuid
from collections import deque


class SurpriseQuestionManager:
    """
    돌발 질문 상태 머신 (단순화).

    상태: IDLE → GENERATING → IDLE
    - 질문 전송 후 즉시 IDLE 복귀 (답변 수집 없음)
    - 답변 평가는 배치 분석 시 STT 텍스트 슬라이싱으로 처리
    트리거: 누적 단어 30개 이상
    """

    WORD_THRESHOLD = 30

    def __init__(self):
        self.state: str = "IDLE"
        self.accumulated_words: int = 0
        self.last_trigger_time: float | None = None
        self.stt_context: deque = deque(maxlen=20)
        self.asked_questions: list[str] = []

    def start(self, presentation_start_time: float):
        self.last_trigger_time = presentation_start_time

    def add_stt(self, text: str):
        if not text or not text.strip():
            return
        self.stt_context.append(text.strip())
        if self.state == "IDLE":
            self.accumulated_words += len(text.split())

    def get_context(self, max_entries: int = 10) -> str:
        return " ".join(list(self.stt_context)[-max_entries:])

    def should_trigger(self, current_time: float) -> bool:
        if self.state != "IDLE":
            return False
        if not self.stt_context:
            return False
        return self.accumulated_words >= self.WORD_THRESHOLD

    def start_generating(self, current_time: float):
        """GPT 질문 생성 시작 — 중복 트리거 방지."""
        self.state = "GENERATING"
        self.last_trigger_time = current_time
        self.accumulated_words = 0

    def set_triggered(self, question: str):
        """질문 전송 완료 — 즉시 IDLE 복귀해서 다음 질문 준비."""
        self.asked_questions.append(question)
        self.state = "IDLE"

    def reset_to_idle(self):
        self.state = "IDLE"
