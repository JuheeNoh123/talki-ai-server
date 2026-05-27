import time
import multiprocessing
import threading

from app.utils.audio_utils import speech_stats


def _to_dict_result(segments_gen):
    """faster-whisper 세그먼트 generator를 speech_stats 호환 dict로 변환"""
    segments = []
    text_parts = []
    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
        segments.append({
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
        })
        text_parts.append(seg.text.strip())
    return {"text": " ".join(text_parts), "segments": segments}


def whisper_worker(conn):
    """Whisper 모델을 로드하고 요청을 처리하는 상주 프로세스"""
    import sys, os
    # spawn된 자식 프로세스는 venv sys.path를 자동 상속하지 못하는 경우가 있음
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        site_pkgs = os.path.join(venv, "Lib", "site-packages")
        if site_pkgs not in sys.path:
            sys.path.insert(0, site_pkgs)

    try:
        print("[Whisper Process] 초기화 시작...")
        init_start = time.time()

        # faster-whisper는 CTranslate2 기반으로 PyTorch 없이 CUDA 사용
        # compute_type="int8": VRAM 절약 (float32 대비 ~75% 감소)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        compute_type = "int8" if device == "cuda" else "int8"
        print(f"[Whisper Process] Device: {device}, compute_type: {compute_type}")

        from faster_whisper import WhisperModel
        try:
            model = WhisperModel("small", device=device, compute_type=compute_type)
        except Exception as cuda_err:
            print(f"[Whisper Process] CUDA 로드 실패 ({cuda_err}), CPU로 전환...")
            device, compute_type = "cpu", "int8"
            model = WhisperModel("small", device=device, compute_type=compute_type)
        init_elapsed = time.time() - init_start
        print(f"[Whisper Process] 모델 로드 완료 ({init_elapsed:.2f}s). 대기 중...")

        while True:
            if not conn.poll(timeout=None):
                continue

            task = conn.recv()
            if task is None:
                break

            audio_path = task
            print(f"[Whisper Process] STT 분석 요청 수신: {audio_path}")

            abs_start = time.time()

            try:
                segments_gen, _ = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    initial_prompt="어, 음, 그, 저, 뭐, 아, 어어, 음음, 그래서, 근데, 어... 음... 그...",
                    language="ko",
                )
                result = _to_dict_result(segments_gen)
                stats = speech_stats(result)

                abs_end = time.time()
                transcribe_elapsed = abs_end - abs_start
                print(f"[Whisper Process] 분석 완료 ({transcribe_elapsed:.2f}s)")

                conn.send({
                    "status": "success",
                    "data": stats,
                    "timing": {
                        "init": init_elapsed,
                        "transcribe": transcribe_elapsed,
                        "abs_start": abs_start,
                        "abs_end": abs_end,
                    },
                })
            except Exception as e:
                print(f"[Whisper Process] 분석 중 에러: {e}")
                conn.send({"status": "error", "message": str(e)})

    except Exception as e:
        print(f"[Whisper Process] 치명적 오류: {e}")
        try:
            conn.send({"status": "fatal_error", "message": str(e)})
        except Exception:
            pass
    finally:
        print("[Whisper Process] 종료")


class WhisperService:
    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=whisper_worker,
            args=(self.child_conn,),
            daemon=True,
        )
        self.started = False
        self._lock = threading.Lock()

    def start(self):
        if not self.started:
            self.process.start()
            self.started = True

    def stop(self):
        if self.started:
            self.parent_conn.send(None)
            self.process.join()
            self.started = False

    def transcribe(self, audio_path):
        with self._lock:
            self.parent_conn.send(audio_path)
            return self.parent_conn.recv()

    def transcribe_async(self, audio_path):
        self.parent_conn.send(audio_path)

    def get_result(self):
        return self.parent_conn.recv()


# 공유 singleton — start()는 FastAPI lifespan에서 호출
whisper_service = WhisperService()
