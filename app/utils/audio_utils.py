import os
import re
from pydub import AudioSegment
from pydub.utils import which


def _ensure_ffmpeg_for_pydub():
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")

    # 시스템 PATH에 없는 경우, 일반적인 설치 경로 확인 및 추가
    if not ffmpeg_path or not ffprobe_path:
        os.environ["PATH"] += os.pathsep + r"C:\\ProgramData\\chocolatey\\bin"
        ffmpeg_path = ffmpeg_path or which("ffmpeg")
        ffprobe_path = ffprobe_path or which("ffprobe")

    # 그래도 없으면 하드코딩된 경로 사용 (사용자 환경에 맞게 수정 필요)
    if not ffmpeg_path:
        ffmpeg_path = r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
    if not ffprobe_path:
        ffprobe_path = r"C:\\ProgramData\\chocolatey\\bin\\ffprobe.exe"

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

def extract_audio(video_path):
    """영상 파일에서 오디오를 추출하여 임시 wav 파일로 저장합니다."""
    import tempfile
    _ensure_ffmpeg_for_pydub()
    audio = AudioSegment.from_file(video_path, format="mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name

def count_fillers_from_words(segs: list) -> int:
    """
    Whisper word_timestamps 기반 필러 감지.

    Whisper는 "어", "음" 등을 텍스트에서 제거하는 대신
    인접 단어의 duration 안으로 흡수시킴.
    예: "어.. 오늘은" → word "오늘은" start=0.0, end=2.4 (3글자인데 2.4초)

    감지 방식: 단어 지속시간 이상 감지
    - 한국어 정상 발화 속도: 글자당 약 0.15~0.25초
    - 글자당 duration이 DURATION_PER_CHAR_THRESHOLD 초 초과 시 필러 흡수로 판단
    - 초과된 시간 / 필러 1개당 평균 길이(0.5초)로 필러 개수 추정
    """
    DURATION_PER_CHAR_THRESHOLD = 0.35  # 글자당 이 초를 넘으면 비정상
    FILLER_AVG_SEC = 0.5                # 필러 1개당 평균 지속시간

    filler_cnt = 0

    for seg in segs:
        words = seg.get("words", [])
        if not words:
            continue

        for word_info in words:
            word = word_info.get("word", "").strip()
            duration = word_info.get("end", 0) - word_info.get("start", 0)
            char_count = max(1, len(word))
            duration_per_char = duration / char_count

            if duration_per_char > DURATION_PER_CHAR_THRESHOLD:
                # 정상 발화로 예상되는 시간 = char_count * 0.2초
                expected_duration = char_count * 0.2
                excess = duration - expected_duration
                # 초과 시간을 필러 평균 길이로 나눠 개수 추정
                estimated = max(1, round(excess / FILLER_AVG_SEC))
                filler_cnt += estimated
                print(f"[Filler] '{word}' duration={duration:.2f}s ({duration_per_char:.2f}s/char) → +{estimated} filler")

    print(f"[Filler] total filler_cnt={filler_cnt}")
    return filler_cnt


def speech_stats(transcribe_result, min_seconds=3.0):
    """
    STT 결과에서 발화 속도(WPM)와 필러(추임새) 사용 빈도를 계산합니다.
    필러는 word_timestamps 기반 pause 감지로 측정 (Whisper가 텍스트에서 제거하므로).

    Returns:
        dict: {text, wpm, cps, duration, fillers_count, fillers_freq,
               silence_count, total_silence_sec, silence_ratio}
    """
    segs = transcribe_result.get("segments", [])
    text = transcribe_result.get("text", "").strip()

    if not segs:
        return {
            "text": text, "wpm": 0.0, "cps": 0.0, "duration": 0.0,
            "fillers_count": 0, "fillers_freq": 0.0,
            "silence_count": 0, "total_silence_sec": 0.0, "silence_ratio": 0.0,
            "avg_word_probability": 0.0,
        }

    start = segs[0]["start"]
    end = segs[-1]["end"]
    dur = max(0.0, end - start)

    # 단어 수 및 글자 수
    words = sum(len(s["text"].split()) for s in segs)
    chars = sum(len(s["text"].replace(" ", "")) for s in segs)

    wpm = (words / (dur / 60.0)) if dur >= min_seconds else 0.0
    cps = (chars / dur) if dur > 0 else 0.0

    # 필러 감지 (두 방식 병행)
    # 1차: initial_prompt 덕에 텍스트에 직접 나온 필러 카운팅 (종류 구분 가능)
    FILLERS = {
        "어": "hesitation",
        "음": "hesitation",
        "그": "filler",
        "저": "filler",
        "뭐": "filler",
        "아": "hesitation",
        "um": "hesitation",
        "uh": "hesitation",
    }
    text_filler_counts = {label: 0 for label in set(FILLERS.values())}
    text_filler_counts["total"] = 0
    filler_list = []  # 등장 순서대로 실제 필러 단어 목록
    for word, label in FILLERS.items():
        pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
        matches = re.findall(pattern, text)
        cnt = len(matches)
        text_filler_counts[label] += cnt
        text_filler_counts["total"] += cnt
        filler_list.extend(matches)

    # 등장 순서 복원: text에서 순서대로 재추출
    ordered_filler_list = []
    filler_pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in FILLERS) + r")(?!\w)"
    for match in re.finditer(filler_pattern, text):
        ordered_filler_list.append(match.group(1))

    # 2차: duration 이상 감지 (텍스트 필러가 0이면 fallback)
    duration_filler_cnt = count_fillers_from_words(segs)

    # 텍스트 기반이 감지되면 우선 사용, 없으면 duration 기반 사용
    if text_filler_counts["total"] > 0:
        filler_cnt = text_filler_counts["total"]
        filler_detail = text_filler_counts
        print(f"[Filler] 텍스트 기반 감지: {filler_detail}, list={ordered_filler_list}")
    else:
        filler_cnt = duration_filler_cnt
        filler_detail = {"total": filler_cnt, "hesitation": 0, "filler": 0}
        ordered_filler_list = []
        print(f"[Filler] duration 기반 감지 fallback: {filler_cnt}")

    filler_freq = (filler_cnt / (dur / 60.0)) if dur >= min_seconds else 0.0

    # 침묵: 세그먼트 사이 gap >= 2초
    SILENCE_GAP_SEC = 2.0
    silence_gaps = []
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i - 1]["end"]
        if gap >= SILENCE_GAP_SEC:
            silence_gaps.append(gap)

    silence_count = len(silence_gaps)
    total_silence_sec = round(sum(silence_gaps), 2)
    silence_ratio = round(total_silence_sec / dur, 3) if dur > 0 else 0.0

    # 단어별 confidence(probability) 평균 — 연습 탭 "발음 명확도" 근사치로 사용
    word_probs = [
        w.get("probability", 0.0)
        for seg in segs
        for w in seg.get("words", [])
        if w.get("probability") is not None
    ]
    avg_word_probability = round(sum(word_probs) / len(word_probs), 3) if word_probs else 0.0

    return {
        "text": text,
        "wpm": float(wpm),
        "cps": float(cps),
        "duration": float(dur),
        "fillers_count": int(filler_cnt),
        "fillers_freq": float(round(filler_freq, 2)),
        "filler_detail": filler_detail,
        "filler_list": ordered_filler_list,  # ex. ["어", "음", "그", "어"]
        "silence_count": silence_count,
        "total_silence_sec": total_silence_sec,
        "silence_ratio": silence_ratio,
        "avg_word_probability": avg_word_probability,
    }
