import time
import multiprocessing
import threading
import torch

from app.utils.audio_utils import speech_stats


def whisper_worker(conn):
    """
    Whisper 모델을 로드하고 요청을 처리하는 상주 프로세스 함수 (Pipe 사용)
    """
    try:
        print("[Whisper Process] 초기화 시작...")
        init_start = time.time()

        # GPU 확인 및 디바이스 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper Process] Device: {device}")

        import whisper
        # 모델 로드 (최초 1회 실행)
        model = whisper.load_model("small", device=device)
        init_elapsed = time.time() - init_start
        print(f"[Whisper Process] 모델 로드 완료 (소요시간: {init_elapsed:.2f}s). 대기 중...")

        while True:
            # Pipe에서 작업 가져오기
            if not conn.poll(timeout=None): # 대기
                continue

            task = conn.recv()
            if task is None: # 종료 신호
                break

            # task는 audio_path
            audio_path = task
            print(f"[Whisper Process] STT 분석 요청 수신: {audio_path}")

            # 절대 시간 기록 (Overlap 계산용)
            abs_start = time.time()

            try:
                # Transcribe
                # initial_prompt: Whisper가 "어", "음" 등 필러를 suppression하지 않고
                # 실제 전사하도록 유도. word_timestamps는 타임스탬프 확보용.
                result = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    initial_prompt="어, 음, 그, 저, 뭐, 아, 어어, 음음, 그래서, 근데, 어... 음... 그..."
                )

                # 통계 계산
                stats = speech_stats(result)

                abs_end = time.time()
                transcribe_elapsed = abs_end - abs_start
                print(f"[Whisper Process] 분석 완료 ({transcribe_elapsed:.2f}s)")

                # 결과 전송
                conn.send({
                    "status": "success",
                    "data": stats,
                    "timing": {
                        "init": init_elapsed,
                        "transcribe": transcribe_elapsed,
                        "abs_start": abs_start,
                        "abs_end": abs_end
                    }
                })
            except Exception as e:
                print(f"[Whisper Process] 분석 중 에러: {e}")
                conn.send({"status": "error", "message": str(e)})

    except Exception as e:
        try:
            conn.send({"status": "fatal_error", "message": str(e)})
        except:
            pass
    finally:
        print("[Whisper Process] 종료")


# Whisper를 독립된 프로세스로 실행
class WhisperService:
    def __init__(self):
        # Queue 대신 메인 프로세스와 통신하기 위한 Pipe 사용 (양방향)
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        # Whisper 작업을 전담할 별도 프로세스 생성
        self.process = multiprocessing.Process(
            target=whisper_worker,
            args=(self.child_conn,),
            daemon=True
        )
        self.started = False
        self._lock = threading.Lock()  # singleton 공유 시 동시 접근 직렬화

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
        """동기 변환 — lock으로 직렬화. singleton 공유 시 이 메서드를 사용."""
        with self._lock:
            self.parent_conn.send(audio_path)
            return self.parent_conn.recv()

    def transcribe_async(self, audio_path):
        """비동기 요청 전송 (결과는 나중에 받음)"""
        self.parent_conn.send(audio_path)

    def get_result(self):
        """결과 수신 대기"""
        return self.parent_conn.recv()


# 공유 singleton — start()는 FastAPI lifespan에서 호출 (모듈 레벨 spawn 방지)
whisper_service = WhisperService()
