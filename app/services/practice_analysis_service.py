# app/services/practice_analysis_service.py
# 연습 탭 4단계(연습 진행) 전용 분석 로직: 시선 고정 누적 통계, 발음 명확도 라벨,
# 키워드 사용 판정, subStep별 0~100 종합 점수, 규칙 기반 피드백 문구.
# 시선 방향 자체 계산(gaze_from_landmarks)은 app/utils/analysis_utils.py를 그대로 재사용한다.

_LABEL_MAP = {"우수": 100, "양호": 80, "보통": 60, "미흡": 40}


class GazeFixationTracker:
    """시선 고정 훈련(GAZE) 세션 동안의 누적 응시 통계."""

    def __init__(self):
        self.total_frames = 0
        self.on_target_frames = 0
        self.break_count = 0
        self._was_on_target = False
        self._hold_start = None
        self._hold_durations: list[float] = []

    def update(self, gaze: dict, now: float) -> bool:
        on_target = gaze.get("horiz") == "center" and gaze.get("vert") == "center"
        self.total_frames += 1

        if on_target:
            self.on_target_frames += 1
            if self._hold_start is None:
                self._hold_start = now
        else:
            if self._hold_start is not None:
                self._hold_durations.append(now - self._hold_start)
                self._hold_start = None
            if self._was_on_target:
                self.break_count += 1

        self._was_on_target = on_target
        return on_target

    def finalize(self, now: float) -> dict:
        if self._hold_start is not None:
            self._hold_durations.append(now - self._hold_start)
            self._hold_start = None

        hold_ratio = round(self.on_target_frames / self.total_frames, 2) if self.total_frames else 0.0
        avg_hold = round(sum(self._hold_durations) / len(self._hold_durations), 1) if self._hold_durations else 0.0

        return {
            "gaze_hold_ratio": hold_ratio,
            "avg_hold_duration_sec": avg_hold,
            "gaze_break_count": self.break_count,
        }


def articulation_label(avg_word_probability: float) -> str:
    """Whisper 단어별 confidence 평균을 발음 명확도 근사 라벨로 변환."""
    if avg_word_probability >= 0.90:
        return "우수"
    if avg_word_probability >= 0.75:
        return "양호"
    if avg_word_probability >= 0.60:
        return "보통"
    return "미흡"


def keyword_usage(keywords: list[str], text: str) -> list[dict]:
    """공백 제거 후 부분 문자열 포함 여부로 키워드 사용을 판정 (형태소 변형은 고려하지 않음)."""
    text_norm = (text or "").replace(" ", "")
    return [
        {"keyword": kw, "used": kw.replace(" ", "") in text_norm}
        for kw in keywords
    ]


# ── 실시간 라이브 피드백 (짧은 문구 배열) ─────────────────────────────────────

def build_script_feedback(wpm: float, fillers_freq: float, wpm_min: int, wpm_max: int) -> list[str]:
    msgs = []
    if wpm > wpm_max:
        msgs.append("말 속도가 조금 빠릅니다.")
    elif 0 < wpm < wpm_min:
        msgs.append("말 속도가 조금 느립니다.")
    if fillers_freq and fillers_freq > 5:
        msgs.append('"음", "아" 같은 추임새가 자주 들립니다.')
    return msgs


def build_gaze_feedback_text(hold_ratio: float) -> str:
    if hold_ratio >= 0.75:
        return "아주 좋습니다! 평균적으로 시선을 잘 유지하고 있습니다."
    if hold_ratio >= 0.5:
        return "시선을 절반 이상 목표 지점에 유지했습니다. 조금 더 연습하면 좋아질 거예요."
    return "시선이 자주 목표 지점을 벗어났습니다. 한 곳을 정해두고 천천히 연습해보세요."


def build_script_final_feedback_text(wpm: float, articulation: str, wpm_min: int, wpm_max: int) -> str:
    if wpm_min <= wpm <= wpm_max:
        speed_desc = "말 속도가 적정 범위에 있습니다"
    elif wpm > wpm_max:
        speed_desc = "말 속도가 다소 빠릅니다"
    else:
        speed_desc = "말 속도가 다소 느립니다"
    return f"{speed_desc}. 발음은 {articulation} 수준입니다."


# ── subStep별 0~100 종합 점수 ─────────────────────────────────────────────────

def calc_script_score(wpm: float, fillers_freq: float, avg_word_probability: float, wpm_min: int, wpm_max: int) -> int:
    if wpm_min <= wpm <= wpm_max:
        wpm_score = 100
    else:
        diff = (wpm_min - wpm) if wpm < wpm_min else (wpm - wpm_max)
        wpm_score = max(0, 100 - diff * 2)
    filler_score = max(0, 100 - fillers_freq * 10)
    articulation_score = round(avg_word_probability * 100)
    return int(round(wpm_score * 0.4 + filler_score * 0.3 + articulation_score * 0.3))


def calc_gaze_score(hold_ratio: float, break_count: int) -> int:
    base = hold_ratio * 100
    penalty = min(30, break_count * 3)
    return int(round(max(0, base - penalty)))


def calc_impromptu_score(spoken_ratio: float, fillers_freq: float, structure_label: str) -> int:
    structure_score = _LABEL_MAP.get(structure_label, 60)
    duration_score = min(100, round(spoken_ratio * 100))
    filler_score = max(0, 100 - fillers_freq * 10)
    return int(round(duration_score * 0.3 + filler_score * 0.2 + structure_score * 0.5))


def calc_keyword_score(used_count: int, total_keywords: int, naturalness_label: str) -> int:
    coverage_score = round((used_count / total_keywords) * 100) if total_keywords else 0
    naturalness_score = _LABEL_MAP.get(naturalness_label, 60)
    return int(round(coverage_score * 0.5 + naturalness_score * 0.5))


def calc_point_score(used_count: int, total_keywords: int, gist_label: str) -> int:
    coverage_score = round((used_count / total_keywords) * 100) if total_keywords else 0
    gist_score = _LABEL_MAP.get(gist_label, 60)
    return int(round(coverage_score * 0.4 + gist_score * 0.6))
