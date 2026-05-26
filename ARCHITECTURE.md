# TALKI AI Server - 아키텍처 문서

## 1. 시스템 개요

TALKI는 발표자의 영상을 분석하여 **음성, 시선, 자세/제스처, 주제 적합성** 4가지 차원에서 평가하고 LLM 기반 코칭 피드백을 제공하는 AI 발표 분석 시스템입니다.

### 운영 모드
| 모드 | 엔드포인트 | 설명 |
|------|-----------|------|
| 배치 분석 | `POST /analyze/record-from-s3` | 녹화된 영상 전체 분석 후 결과 콜백 |
| 실시간 분석 | `WebSocket /realtime` | 발표 중 실시간 이상 감지 및 피드백 스트리밍 |

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        클라이언트 (앱)                            │
└────────────────────┬────────────────────┬───────────────────────┘
                     │ HTTP               │ WebSocket
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Spring Boot 서버 (외부)                         │
│  - 사용자 인증 / 발표 관리                                          │
│  - /analyze/callback 수신                                         │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP (S3 URL + 메타데이터)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastAPI 서버 (본 레포)                            │
│                                                                  │
│  ┌─────────────────┐      ┌──────────────────────────────────┐  │
│  │  analyze_router │      │         realtime_router           │  │
│  │  POST /analyze  │      │   WebSocket /realtime?type={}     │  │
│  └────────┬────────┘      └──────────────┬───────────────────┘  │
│           │                              │                        │
│           ▼                              ▼                        │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ analyze_service     │  │  analyze_service_landmarks       │   │
│  │ _optimized          │  │  + feedback_manager              │   │
│  └──────────┬──────────┘  └──────────────┬──────────────────┘   │
│             │                             │                        │
│             ▼                             ▼                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              핵심 분석 파이프라인                            │    │
│  │  test_record_multiprocess.py (멀티프로세스)                  │    │
│  │  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │ MediaPipe │ │ MediaPipe │ │ Whisper  │ │  Topic   │  │    │
│  │  │ FaceMesh  │ │   Pose    │ │   STT    │ │  Model   │  │    │
│  │  │ (시선분석) │ │(자세분석)  │ │(음성분석) │ │(주제분석) │  │    │
│  │  └───────────┘ └───────────┘ └──────────┘ └──────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│             │                             │                        │
│             ▼                             ▼                        │
│  ┌──────────────────┐         ┌──────────────────────────┐       │
│  │  feedback_service│         │    Redis (세그먼트 저장)    │       │
│  │  점수 계산 +       │         │  실시간 이벤트 타임라인     │       │
│  │  GPT-4.1 피드백   │         └──────────────────────────┘       │
│  └──────────┬───────┘                                             │
│             │ HTTP Callback                                         │
└─────────────┼───────────────────────────────────────────────────┘
              ▼
       Spring Boot /analyze/callback
```

---

## 3. 디렉터리 구조

```
talki-ai-server/
├── app/
│   ├── main.py                          # FastAPI 앱 초기화, 라우터 등록
│   ├── config/
│   │   └── feedback_criteria.py         # 발표 유형별 점수 기준 임계값
│   ├── core/
│   │   └── redis.py                     # Redis 클라이언트 설정
│   ├── routers/
│   │   ├── analyze_router.py            # POST /analyze/record-from-s3
│   │   └── realtime_router.py           # WebSocket /realtime
│   ├── schemas/
│   │   └── analyze_schema.py            # Pydantic 요청/응답 모델
│   ├── services/
│   │   ├── analyze_service_optimized.py # 배치 분석 오케스트레이터
│   │   ├── analyze_service_landmarks.py # 실시간 랜드마크 분석
│   │   ├── feedback_service.py          # 점수 계산 & 피드백 생성
│   │   ├── feedback_manager.py          # 실시간 피드백 상태 관리
│   │   └── whisper_service.py           # 음성→텍스트 서비스 래퍼
│   ├── llm/
│   │   ├── hf_model.py                  # OpenAI GPT-4.1-mini 연동
│   │   └── prompt_builder.py            # LLM 프롬프트 생성
│   └── utils/
│       └── analysis_utils.py            # 시선/자세 계산 유틸
├── Topic_model/
│   ├── service_scorer.py                # 주제 관련성 점수 계산
│   ├── topic_model_mnr/                 # KoSimCSE 사전학습 모델
│   └── label_model/                     # 문장 품질 판별 모델
├── test_record_multiprocess.py          # 핵심 병렬 분석 파이프라인
├── test_record_lazy.py                  # 프레임/오디오 처리 유틸
├── requirements.txt
└── .env
```

---

## 4. 배치 분석 흐름 (POST /analyze/record-from-s3)

```
클라이언트 요청 (video_url, type, topic)
         │
         ▼
  [analyze_router]
  S3에서 영상 다운로드 → 임시 파일 저장
  Semaphore(2) — 동시 작업 최대 2개
         │
         ▼
  [analyze_service_optimized]
  analyze_parallel() 호출 — 멀티프로세싱
         │
    ┌────┴────────────────────────────────┐
    ▼                                     ▼
  영상 분석 (MediaPipe)              음성 분석 (Whisper)
  - 프레임 스트라이드 추출             별도 프로세스 (Pipe IPC)
  - 시선 방향 히스토그램               - WPM 계산
  - 손/팔 움직임 속도                  - 필러워드 감지 (음, 어)
  - 자세 경직/불안정 감지              - 침묵 구간 분석
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
           [ServiceScorer]
           주제 적합성 분석 (KoSimCSE)
                   │
                   ▼
           [feedback_service]
           점수 계산 (0~100):
           총점 = 0.5×(시선+자세)/2
                + 0.3×(음성+필러)/2
                + 0.2×주제
                   │
                   ▼
           [hf_model / prompt_builder]
           GPT-4.1-mini → 한국어 피드백 JSON
           (9개 카테고리)
                   │
                   ▼
           Spring Boot /analyze/callback
           HTTP POST 결과 전송
```

---

## 5. 실시간 분석 흐름 (WebSocket /realtime)

```
클라이언트 WebSocket 연결 (?type=발표유형)
         │
         ▼  매 프레임 전송 (base64 이미지 + 오디오)
  [realtime_router]
         │
         ├──▶ [analyze_service_landmarks]
         │    - MediaPipe FaceMesh → 시선 (dx, dy)
         │    - MediaPipe Pose → 손/팔 위치
         │
         ├──▶ [feedback_manager]
         │    - 롤링 버퍼로 WPM/필러/시선/자세 축적
         │    - 기준치 초과 시 피드백 메시지 생성
         │    - 5초 쿨다운 (스팸 방지)
         │
         ├──▶ Whisper STT (5초마다 배치)
         │    - 누적 오디오 버퍼 → 텍스트
         │    - 발화 속도 이상 감지
         │
         ▼
  감지 이벤트 → Redis 저장 (TTL 1시간)
  키: presentation:{id}:segments

  세그먼트 유형:
  - speech_fast / speech_slow  : 발화 속도
  - silence                    : 3초 이상 침묵
  - pose_rigid / pose_unstable : 자세 경직/과도한 움직임
  - gaze_unstable              : 불안정한 시선

         │
         ▼  실시간 피드백 JSON 스트리밍
  클라이언트 (발표 중 코칭)
```

---

## 6. 점수 산출 방식

### 발표 유형별 기준 (feedback_criteria.py)

| 지표 | 온라인 소규모 | 소규모 | 대규모 |
|------|------------|--------|--------|
| WPM 범위 | 80–110 | 80–110 | 80–110 |
| 필러 허용 (회/분) | 2 | 3 | 4 |
| 자세 최소 속도 | 0.010 | 0.012 | 0.014 |
| 자세 최대 속도 | 0.018 | 0.020 | 0.022 |
| 시선 정면 비율 | 0.65 | 0.60 | 0.55 |

### 총점 가중치

```
총점 = 0.5 × (시선점수 + 자세점수) / 2
     + 0.3 × (음성점수 + 필러점수) / 2
     + 0.2 × 주제점수
```

---

## 7. LLM 피드백 (GPT-4.1-mini)

`prompt_builder.py`가 원시 수치를 자연어 해석으로 변환 후 GPT에 전달합니다.

**출력 JSON 9개 필드:**
| 필드 | 설명 |
|------|------|
| 장점 | 잘한 점 |
| 성장 포인트 | 개선이 필요한 점 |
| 연습 | 연습 권장사항 |
| 음성 분석 결과 | WPM·침묵·발화 속도 |
| 반복어 분석 결과 | 필러워드 빈도 및 목록 |
| 시선 분석 결과 | 시선 방향 분포 |
| 자세/제스처 분석 결과 | 손/팔 움직임 패턴 |
| 주제 적합성 분석 결과 | 내용 일관성 평가 |
| 전체 분석 결과 | 종합 평가 |

---

## 8. 주요 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 프레임워크 | FastAPI, Uvicorn |
| 컴퓨터 비전 | MediaPipe (FaceMesh 468점, Pose 33점), OpenCV |
| 음성 인식 | OpenAI Whisper |
| NLP / 주제 분석 | KoSimCSE (sentence-transformers), RoBERTa |
| LLM 피드백 | OpenAI GPT-4.1-mini |
| 실시간 저장 | Redis (세그먼트 이벤트 타임라인) |
| 병렬 처리 | Python multiprocessing (spawn) |
| 스키마 검증 | Pydantic v2 |

---

## 9. 환경 변수 (.env)

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | GPT-4.1-mini 호출 |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASS` | Redis 연결 |
| `SPRING_URL` | 배치 분석 결과 콜백 주소 |
| `TOPIC_MODEL_DIR` | 사전학습 모델 경로 |
