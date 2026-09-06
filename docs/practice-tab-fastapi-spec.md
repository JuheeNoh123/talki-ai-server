# 연습 탭(CBT 6단계) — FastAPI 담당분(4단계: 연습 진행) API 명세서

> 작성 기준: 실전 탭 기존 구현(`app/routers/analyze_router.py`, `app/routers/realtime_router.py` 등) 코드 조사 결과 +
> Spring 쪽에서 이미 확정한 "연습 탭 API 명세서 — Spring 담당분" 문서(4절 "Spring → FastAPI 인터페이스 계약").
> **이 문서는 Spring 문서의 4절 계약을 그대로 준수하는 것을 최우선 제약으로 삼고, 그 안에서 FastAPI 내부 구현(재사용/신규)을 설계한다.**
> Spring 문서와 맞지 않거나 비어 있는 부분은 "⚠️ Spring 문서와 정합성 확인 필요"로 별도 표기했다(§7).

통신 방식 관련 이미 확인된 결정 사항:
- 스크립트 읽기 / 즉흥 구성 3종의 STT(WPM·필러워드) 분석 → **WebSocket 스트리밍**(Spring 문서의 `/practice/realtime` 단일 엔드포인트 재사용, 신규 로직은 STT 버퍼링 주기 처리).
- 시선 고정 훈련의 랜드마크 처리 → **클라이언트가 MediaPipe FaceMesh로 좌표를 뽑아 WS로 전송, 서버는 계산만**(실전 탭과 동일 패턴, 그대로 재사용).
- **엔드포인트는 Spring 문서와 동일하게 `WS /practice/realtime?sessionId={id}&subStep={type}` 단일 엔드포인트로 통일**한다. 즉흥 구성 연습의 주제/키워드 생성(LLM)도 별도 REST가 아니라, WS 연결 직후 FastAPI가 즉시 생성해 `session_start` 메시지에 담아 보낸다.

---

## 0. 구현 현황 (완료)

이 문서에 정의된 내용은 실제로 구현 완료되어 로컬 GPU PC의 Docker Desktop 컨테이너(`juheenoh/talki-ai-server:latest`)로 배포되었고, Spring이 붙어서 테스트할 수 있는 상태다.

**구현 파일**
| 파일 | 역할 |
|---|---|
| `app/routers/practice_router.py` | `WS /practice/realtime` 엔드포인트 본체 (session_start → 스트리밍 → feedback → result) |
| `app/config/practice_criteria.py` | subStep 상수, 스크립트 풀, WPM 기준값, 시간(prep/speak/target) 설정 |
| `app/services/practice_llm_service.py` | 주제/키워드/원문 생성 + 정성 평가 LLM 호출 (실패 시 fallback 값 반환) |
| `app/services/practice_analysis_service.py` | `GazeFixationTracker`, 발음 명확도 라벨링, 키워드 사용 판정, subStep별 0~100 점수 계산, 규칙 기반 피드백 문구 |
| `app/llm/practice_prompts.py` | LLM 프롬프트 빌더 (§6과 동일) |
| `app/utils/audio_utils.py` (수정) | `speech_stats()`에 `avg_word_probability` 필드 추가 (발음 명확도 근사치 산출용, 기존 필드는 그대로 유지) |
| `app/main.py` (수정) | `practice_router` 등록 |

**§7에서 "정합성 확인 필요"로 남겨뒀던 4가지는 아래와 같이 확정해서 구현했다** (Spring과 최종 협의는 아직 필요):
1. `POINT`의 `reference_keywords` — `session_start`에 포함해서 전송하도록 구현.
2. 하위단계 종료 트리거 — 클라이언트의 첫 오디오/랜드마크 프레임 수신 시점을 기준으로, `speak_seconds`(또는 `target_duration_sec`) 경과 시 FastAPI가 스스로 `result`를 보내고 연결을 닫는 방식으로 구현. 클라이언트가 중간에 연결을 끊어도 그 시점까지 모은 데이터로 best-effort 결과를 계산해 전송을 시도.
3. 라이브 피드백(`feedback.data`) — Spring 문서 예시대로 문자열 배열만 사용 (수치 필드는 넣지 않음). 실시간 진행바에 숫자가 필요하면 추후 협의.
4. `score`(0~100) 산출 — subStep별 가중합 방식으로 신규 구현 (`practice_analysis_service.calc_*_score`).

**검증한 내용**
- 문법/단위 테스트: `practice_analysis_service.py`의 순수 함수(시선 추적, 점수 계산, 피드백 문구) 단독 실행 검증.
- `fastapi.testclient.TestClient` 기반으로 5개 subStep(`SCRIPT`/`GAZE`/`IMPROMPTU`/`KEYWORD`/`POINT`) 전체 흐름을 실제 whisper_service(로컬 GPU) + 고정 LLM 응답(monkeypatch, 실제 API 비용 없음)으로 엔드투엔드 실행 — 52개 assertion 전부 통과.
- Docker로 빌드·배포한 실제 컨테이너에 진짜 WebSocket 클라이언트로 `GAZE`/`SCRIPT` 붙여서 재검증 — 정상 동작 확인.

**아직 검증 못 한 것 / 남은 일**
- 실제 브라우저 MediaPipe 좌표·실제 마이크 음성으로는 미검증 (전부 합성 데이터로 테스트).
- `IMPROMPTU`/`KEYWORD`/`POINT`의 실제 LLM 응답 품질(생성/정성평가 모두)은 미검증 — 첫 실사용 시 실제 OpenAI 과금 발생.
- Spring 쪽 실제 연동(WS 중계, Redis 세션 관리, 결과 저장)은 Spring 팀 테스트 대기 중.

---

## 1. 연습 탭 4단계(연습 진행) 개요 및 FastAPI 담당 범위

6단계 CBT 위저드(Spring 문서 기준 0~7단계) 중 **4단계 "연습 진행"의 AI 분석만 FastAPI 담당**이다. Spring이 클라이언트 WS ↔ FastAPI WS를 그대로 중계(`FastApiWebSocketClient`)하고, FastAPI는 하위단계(`subStep`)별로 실시간 분석과 최종 결과 산출을 수행한다.

| subStep | 하위 단계 | FastAPI 역할 |
|---|---|---|
| `SCRIPT` | 스크립트 읽기 연습 | 낭독 음성을 실시간 STT하여 WPM·필러·발음 명확도 산출 |
| `GAZE` | 시선 고정 훈련 | 목표 지점 대비 응시 정확도(유지율/이탈 횟수) 산출 |
| `IMPROMPTU` | 즉흥 말하기 | LLM으로 주제 질문 생성(연결 시 즉시) + 발화 STT 분석 |
| `KEYWORD` | 키워드 기반 구성 | LLM으로 키워드 3개 생성(연결 시 즉시) + 발화 내 키워드 사용/연결 분석 |
| `POINT` | 핵심 파악 | LLM으로 원문+참고 키워드 생성(연결 시 즉시) + 발화 요약 정확도 분석 |

세션/위저드 전체 상태(어느 단계까지 진행했는지, `subStepQueue`, 최종 리포트 저장)는 Spring이 Redis(진행 중)·MySQL(완료 시)로 관리한다. **FastAPI는 WS 연결이 살아있는 동안에만 해당 하위단계의 분석 상태를 메모리에 들고, 연결이 끝나면 소멸시키는 완전 stateless 서비스**로 설계한다 — 실전 탭처럼 `presentation:{id}:*` Redis 키를 쓰지 않는다(하위단계 하나당 소요가 수십 초~수 분이고, 종료 시 `result` 메시지 한 번으로 Spring에 결과를 넘기면 그걸로 FastAPI 쪽 책임은 끝나기 때문).

**가정(assumption)**:
- `sessionId`는 Spring이 발급해 WS 연결 시 query parameter로 전달한다(실전 탭처럼 FastAPI가 `uuid4()`로 발급하지 않음). FastAPI는 이 값을 로깅/식별 용도로만 사용하고 별도 저장은 하지 않는다.
- 하위단계 종료 시점은 **FastAPI가 각 subStep의 목표 시간(발화 시간/응시 목표 시간)이 다 찼다고 판단하면 스스로 `result`를 보내고 연결을 닫는 방식**으로 설계한다(클라이언트가 별도 "종료" 신호를 보내지 않아도 됨). 사용자가 화면의 "◀ ▶"로 조기 스킵/이탈하면 Spring 또는 클라이언트가 WS를 먼저 끊을 수 있고, 이 경우 FastAPI는 `WebSocketDisconnect` 시점까지 모은 데이터로 best-effort 최종 결과를 계산해 전송을 시도한 뒤 정리한다(실전 탭 `realtime_router.py`의 `finally` 블록 패턴 재사용).
- 각 하위단계 종료 시 언급된 "짧은 구간 녹화"는 클라이언트 재생/보관용이며, FastAPI가 영상 파일 자체를 받아 배치 분석할 필요는 없다(§4 참고) — 오디오·랜드마크가 이미 실시간으로 서버에 도달했으므로 영상 재분석은 중복.
- "발음 명확도"는 기존 코드에 없는 지표라 Whisper 단어별 confidence(`word.probability`, `whisper_service.py`의 `_to_dict_result`에서 이미 추출되지만 현재 미사용)를 근사치로 활용한다. 정확한 조음 평가가 아니라 "STT가 얼마나 확신을 갖고 인식했는가"의 proxy임을 공유 필요.
- `KEYWORD`/`POINT`의 세부 채점 기준(연결 자연스러움, 핵심 파악 정확도)은 확정 스펙이 없어 LLM 정성 평가로 설계했다.

---

## 2. FastAPI가 Spring에 제공하는 엔드포인트

| # | 프로토콜 | Path | 용도 |
|---|---|---|---|
| 1 | WebSocket | `/practice/realtime?sessionId={id}&subStep={SCRIPT\|GAZE\|IMPROMPTU\|KEYWORD\|POINT}` | 하위단계 실시간 분석 (Spring 문서 §4와 동일 경로) |

REST 엔드포인트는 없다. 즉흥 구성 연습의 주제/키워드/원문 생성은 별도 REST 호출이 아니라 **WS 연결 수립 직후 FastAPI가 LLM을 호출해 `session_start` 메시지에 실어 보낸다** — Spring이 하위단계 진입 시점에 정확히 한 번만 WS를 여는 기존 구조와 맞추기 위함이다. (§7에 지연 시간 관련 유의사항 기재)

---

## 3. Request/Response 스키마

Spring 문서 §4의 envelope 규칙을 그대로 따른다: 제어 필드(`type`, `subStep`)는 그대로, AI 산출 지표 필드는 snake_case.

### 3.1 연결

```
ws://{host}/practice/realtime?sessionId={sessionId}&subStep={SCRIPT|GAZE|IMPROMPTU|KEYWORD|POINT}
```

### 3.2 세션 시작 — FastAPI → Spring → Client (연결 수립 직후, 1회)

공통: `{ "type": "session_start", "subStep": "..." }` + subStep별 추가 필드.

**`SCRIPT`** (LLM 미사용, 사전 정의된 스크립트 풀에서 선택)
```json
{
  "type": "session_start",
  "subStep": "SCRIPT",
  "script_text": "안녕하세요. 오늘은 짧지만 중요한 이야기를 해보려고 합니다...",
  "speak_seconds": 60,
  "reference_range": { "wpm_min": 120, "wpm_max": 160 }
}
```

**`GAZE`** (LLM 미사용)
```json
{
  "type": "session_start",
  "subStep": "GAZE",
  "target_duration_sec": 30
}
```

**`IMPROMPTU`** (LLM 호출, §6.1 프롬프트)
```json
{
  "type": "session_start",
  "subStep": "IMPROMPTU",
  "topic": "최근에 배운 것 중 가장 유용했던 것은 무엇인가요?",
  "prep_seconds": 10,
  "speak_seconds": 30
}
```

**`KEYWORD`** (LLM 호출, §6.2 프롬프트)
```json
{
  "type": "session_start",
  "subStep": "KEYWORD",
  "keywords": ["협업", "문제해결", "성장"],
  "prep_seconds": 10,
  "speak_seconds": 30
}
```

**`POINT`** (LLM 호출, §6.3 프롬프트)
```json
{
  "type": "session_start",
  "subStep": "POINT",
  "passage": "효과적인 팀워크를 위해서는 무엇보다도 명확하고 원활한 의사소통이 필수적입니다...",
  "reference_keywords": ["의사소통", "경청", "협력"],
  "speak_seconds": 30
}
```
> ⚠️ Spring 문서 §4.2 표에는 `POINT`의 추가 필드가 `passage`만 명시되어 있으나, 화면 상 "이 단어들을 포함시켜 답변해보세요(참고용)" 칩이 표시되므로 `reference_keywords`가 실제로는 필요하다고 판단해 포함했다. Spring 스키마에도 이 필드 반영이 필요하다(§7).

### 3.3 클라이언트 → FastAPI (Spring이 그대로 패스스루) — 실시간 스트리밍

```json
{ "audio": "BASE64_PCM16_16K_MONO", "timestamp": 1712345678123 }
```
```json
{ "face": { "468": {"x":0.42,"y":0.51}, "469": {"x":0.43,"y":0.50} }, "timestamp": 1712345678123 }
```
- 음성 위주 subStep(`SCRIPT`, `IMPROMPTU`, `KEYWORD`, `POINT`): `audio` + `timestamp`. `type` 필드 없음(기존 `/realtime`과 동일 컨벤션 — `analyze_service_landmarks.py`가 `data.get("audio")`/`data["face"]`로 바로 읽는 방식 재사용).
- `GAZE`: `face` + `timestamp`. 필수 인덱스는 `analysis_utils.REQUIRED_FACE_IDX` 그대로 사용.

### 3.4 진행 중 라이브 피드백 — FastAPI → Spring → Client (주기적, 여러 번)

```json
{ "type": "feedback", "subStep": "SCRIPT", "data": ["말 속도가 조금 빠릅니다."] }
```
`data`는 화면에 바로 노출 가능한 짧은 한국어 문장 배열(0개 이상). 세부 수치(WPM 등)는 라이브 피드백에는 담지 않고 `result`의 `raw_result`에서 한 번에 전달한다 — 스크린샷 상 "실시간 분석" 패널의 진행바(말 속도/발음 명확도)는 Spring/클라이언트가 `raw_result` 누적치가 아니라 이 `feedback.data` 문구 + 최근 `result` 수치를 조합해 그리는 것으로 가정.

> ⚠️ 진행 중 수치(WPM 진행바 등)를 매 구간 갱신하려면 `feedback` 메시지에도 수치 필드가 필요할 수 있다. Spring 문서 예시가 문자열 배열만 규정하고 있어 우선 이를 따르되, 프론트가 실시간 숫자 표시를 요구하면 `data`를 문자열 배열 대신 `{ "messages": [...], "metrics": {...} }` 형태로 확장하는 협의가 필요하다(§7).

### 3.5 하위단계 종료 시 최종 결과 — FastAPI → Spring → Client (1회, 이후 연결 종료)

```json
{
  "type": "result",
  "subStep": "SCRIPT",
  "score": 82,
  "feedback_text": "전반적으로 안정적인 속도였어요.",
  "raw_result": {
    "wpm": 142.0,
    "fillers_count": 3,
    "fillers_freq": 6.4,
    "duration_sec": 28.0,
    "articulation_label": "양호",
    "text": "인식된 전체 발화 텍스트..."
  }
}
```

subStep별 `raw_result` 추가 필드:

**`GAZE`**
```json
{
  "gaze_hold_ratio": 0.78,
  "avg_hold_duration_sec": 3.2,
  "gaze_break_count": 4
}
```

**`KEYWORD`**
```json
{
  "keyword_usage": [
    { "keyword": "협업", "used": true },
    { "keyword": "문제해결", "used": false },
    { "keyword": "성장", "used": true }
  ],
  "keyword_coverage": "2/3",
  "connection_naturalness": "양호"
}
```

**`POINT`**
```json
{
  "reference_keyword_coverage": "2/3",
  "gist_accuracy_label": "양호",
  "missing_points": ["협력"]
}
```

**`IMPROMPTU`**
```json
{
  "filler_count_label": 3,
  "spoken_duration_sec": 28,
  "structure_completeness": "양호"
}
```

`score`(0~100, `PracticeSubStepResult.score`용)와 `feedback_text`(한 줄 요약)는 모든 subStep 공통 필드다. 산출 방식은 §5 참고.

전송 직후 FastAPI가 WebSocket 종료(CLOSE frame)를 보낸다 — Spring 문서 §4.5 가정과 동일.

---

## 4. 실시간 vs 배치 처리 구분 및 처리 구조

전 구간이 **실시간 WebSocket 스트리밍**이며, 실전 탭의 배치(REST, 영상 전체 후처리) 방식은 연습 탭에 없다.

| 구간 | 처리 구조 |
|---|---|
| 연결 직후 LLM 생성(`IMPROMPTU`/`KEYWORD`/`POINT`) | `run_in_executor`로 OpenAI 호출을 블로킹 실행 후 `session_start` 전송. 응답 전까지 클라이언트는 "준비 중" 상태로 대기(약 1~2초 예상). |
| 오디오 스트리밍(`SCRIPT`/`IMPROMPTU`/`KEYWORD`/`POINT`) | 실전 탭 `realtime_router.py`의 오디오 버퍼링 패턴(청크 누적 → 5초 주기 `_write_wav` → `whisper_service.transcribe`를 `run_in_executor`로 블로킹 호출) **그대로 재사용**. Whisper는 기존 상주 프로세스(`whisper_service`, Pipe IPC) 공유 — 별도 프로세스 추가 기동 없음. `speak_seconds` 경과 시 버퍼 전체를 이어붙여 `speech_stats`를 한 번 더 돌려 최종 결과 산출. |
| 랜드마크 스트리밍(`GAZE`) | 실전 탭 `analyze_service_landmarks.analyze_realtime_landmarks` / `analysis_utils.gaze_from_landmarks` **그대로 재사용**. MediaPipe는 서버에서 실행하지 않음(클라이언트가 랜드마크만 전송) — 서버 CPU/GPU 부하 없음. `target_duration_sec` 동안의 누적 통계(hold_ratio 등)만 신규 로직. |

**멀티프로세싱(`test_record_multiprocess.py`) 기반 영상 배치 분석은 연습 탭에 재사용하지 않는다.** 그 파이프라인은 "서버가 영상 파일을 받아 프레임 단위로 MediaPipe를 직접 실행"하는 구조인데, 연습 탭은 클라이언트가 랜드마크·오디오를 이미 실시간 스트리밍하므로 서버가 영상을 다시 받아 재처리할 이유가 없다.

---

## 5. 기존 실전 탭 모듈 재사용 vs 신규 구분표

| 기능 | 기존 코드 위치 | 재사용 여부 | 비고 |
|---|---|---|---|
| Whisper STT 실행(상주 프로세스, Pipe IPC) | `app/services/whisper_service.py` (`whisper_service.transcribe`) | **100% 재사용** | 신규 프로세스 기동 없이 기존 싱글톤 공유 |
| WPM/필러 계산 | `app/utils/audio_utils.py` (`speech_stats`, `count_fillers_from_words`) | **100% 재사용** | 순수 함수라 오디오만 있으면 호출 가능 |
| 오디오 청크 버퍼링/WAV 변환 패턴 | `app/routers/realtime_router.py` (`audio_buffer`, `_write_wav`, `STT_INTERVAL`) | **100% 재사용** | presentation-type 임계값 분기(`FeedbackManager`)는 제거하고 골격만 이식 |
| 시선 방향 계산 | `app/utils/analysis_utils.py` (`gaze_from_landmarks`) | **100% 재사용** | |
| 랜드마크 수신 포맷/필수 인덱스 | `app/utils/analysis_utils.py` (`REQUIRED_FACE_IDX`) | **100% 재사용** | 클라이언트 계약 동일 유지 |
| 발음 명확도(articulation) 산출 | 없음(단, `whisper_service.py`의 `_to_dict_result`가 `word.probability`를 이미 추출) | **부분 재사용 + 신규** | word.probability 평균 → 라벨 매핑하는 신규 함수 1개 추가 필요 |
| 시선 유지율/이탈 횟수 누적 집계 | 없음(실전 탭은 threshold 초과 여부만 실시간 피드백, 누적 통계 없음) | **신규** | 단순 카운터 수준의 `GazeFixationTracker` 신규 필요 |
| WPM 판정 기준값(120–160 등) | 없음(기존 `FEEDBACK_CRITERIA`는 80–110, 발표 시나리오용) | **신규 설정값** | `app/config/practice_criteria.py` 신규 — "낭독"은 발표 대비 기준 범위가 다름 |
| LLM 호출 클라이언트/패턴(system+assistant 예시+user, JSON 강제) | `app/llm/hf_model.py` (`translate_to_korean`) | **패턴만 재사용** | `client.responses.create` 호출 구조는 동일, 프롬프트 내용은 신규(§6) |
| 짧은 텍스트 LLM 생성(질문 등) 패턴 | `app/services/surprise_question_service.py` (`generate_question`) | **패턴만 재사용** | 주제/키워드/원문 생성용 신규 함수로 변형, 실행 시점을 WS 연결 직후로 이동 |
| 키워드 사용 여부/커버리지 판정 | 없음 | **신규** | 우선 단순 포함 검사, 필요시 형태소 분석 고도화 |
| 연결 자연스러움/핵심 파악 정확도 LLM 평가 | 없음(`feedback_service.py`의 점수화 패턴 참고 가능) | **신규(패턴만 참고)** | §6.4 참고 |
| subStep별 종합 `score`(0~100) 산출 | 없음(실전 탭은 항목별 점수만, 단일 종합 점수 없음) | **신규** | `feedback_service.calc_*_score` 스타일의 0~100 스케일 함수를 subStep별로 1개씩 정의 후 그대로 사용/가중합 |
| 세션ID 발급/Redis 상태 관리 | `app/routers/realtime_router.py` (`presentation_id`, `presentation:{id}:*`) | **미사용** | 연습 탭은 Spring이 `sessionId` 발급, FastAPI는 WS 연결 생존 기간만 상태 유지 |
| 영상 프레임 멀티프로세싱 배치 분석 | `test_record_multiprocess.py` | **미사용** | §4 참고 |

---

## 6. LLM 프롬프트 설계 초안 (주제/키워드 생성용)

기존 `app/llm/hf_model.py`의 호출 패턴(OpenAI `client.responses.create`, `model="gpt-4.1-mini"`, system 규칙 + assistant few-shot 예시 + user 입력, JSON 강제)을 그대로 따른다. 3개 모두 WS 연결 수립 시 동기 호출되어 `session_start`에 실린다.

### 6.1 즉흥 말하기 — 질문 생성 (`subStep=IMPROMPTU`)

```
[system]
너는 발표 불안 완화 훈련을 설계하는 코치다.
사용자가 10초 준비 후 30초간 즉흥으로 답변할 수 있는, 부담스럽지 않고 일상적인 질문을 1개 생성하라.
조건:
- 개인 경험이나 의견을 묻는 형태(예: "최근에 ~한 경험은?", "~에 대해 어떻게 생각하나요?")
- 논쟁적이거나 민감한 주제(정치/종교/개인정보) 금지
- 30초 안에 답변 가능한 난이도
- 반드시 아래 JSON 형식으로만 출력:
{ "question": "" }

[assistant 예시]
{ "question": "최근에 배운 것 중 가장 유용했던 것은 무엇인가요?" }

[user]
(이전에 출제된 질문 목록이 있다면 중복 방지를 위해 전달: 예) 이미 사용된 질문: ["최근에 배운 것 중..."]
```

### 6.2 키워드 기반 구성 — 키워드 3개 생성 (`subStep=KEYWORD`)

```
[system]
너는 발표 구성력 훈련을 설계하는 코치다.
서로 자연스럽게 하나의 이야기로 연결할 수 있는 한국어 키워드 3개를 생성하라.
조건:
- 키워드는 명사형 단어 2~5글자
- 지나치게 추상적이거나 서로 무관한 조합 금지(예: "우주"+"김치"+"회계")
- 직장/학교/일상 경험과 자연스럽게 엮일 수 있는 소재 우선
- 반드시 아래 JSON 형식으로만 출력:
{ "keywords": ["", "", ""] }

[assistant 예시]
{ "keywords": ["협업", "문제해결", "성장"] }

[user]
(난이도나 도메인 힌트가 있다면 전달, 없으면 빈 문자열)
```

### 6.3 핵심 파악 — 원문 + 참고 키워드 생성 (`subStep=POINT`)

```
[system]
너는 발표 핵심 요약 훈련을 설계하는 코치다.
200~300자 분량의 한국어 글 1개와, 그 글의 핵심을 담은 참고 키워드 3개를 생성하라.
조건:
- 글은 2문단 이내, 팀워크/커뮤니케이션/자기계발 등 무난한 소재
- 참고 키워드 3개는 글의 핵심 메시지를 요약했을 때 반드시 언급되어야 하는 단어
- 반드시 아래 JSON 형식으로만 출력:
{ "source_text": "", "reference_keywords": ["", "", ""] }

[assistant 예시]
{
  "source_text": "효과적인 팀워크를 위해서는...",
  "reference_keywords": ["의사소통", "경청", "협력"]
}

[user]
(난이도 힌트, 없으면 빈 문자열)
```

### 6.4 (참고) 발화 종료 후 정성 평가 프롬프트 — 골격만

3.5절의 `connection_naturalness` / `gist_accuracy_label` / `structure_completeness` 산출용으로, `feedback_service.py`의 "점수 계산 → 태그 구성 → LLM 호출" 흐름을 참고해 아래처럼 짧게 설계할 수 있다(세부 임계값/문구는 추후 확정):

```
[system]
너는 말하기 연습 코치다. 아래 정보를 보고 두 가지만 평가해 JSON으로 답하라.
- keyword_coverage: 몇 개 키워드가 자연스럽게 쓰였는지(0/1/2/3)
- naturalness_label: "우수"|"양호"|"보통"|"미흡" 중 하나(키워드 간 연결이 억지스럽지 않은 정도)
{ "keyword_coverage": 0, "naturalness_label": "" }

[user]
키워드: ["협업","문제해결","성장"]
발화 텍스트: "{STT 결과 text}"
```

이 최종 평가 프롬프트는 요구사항의 "주제/키워드 생성용" 범위를 벗어나므로 골격만 제시했고, 세부 설계는 별도 확정이 필요하다.

---

## 7. Spring 문서와의 정합성 확인 필요 사항

Spring이 이미 문서화한 계약과 비교했을 때, 아래 항목은 이번 설계에서 FastAPI 쪽 필요에 의해 보완했거나 확인이 필요하다.

1. **`POINT`의 `reference_keywords` 필드** — Spring 문서 §4.2 표에는 없지만, 화면상 "참고용" 키워드 칩이 필요해 `session_start`에 추가했다(§3.2). Spring 쪽 `session_start` 파싱 로직에도 반영 필요.
2. **하위단계 종료 트리거** — Spring 문서도 "⚠️ 가정"으로 남겨둔 부분과 동일하게, 이 문서는 "FastAPI가 `speak_seconds`/`target_duration_sec` 경과를 스스로 판단해 종료"로 확정해 설계했다. Spring이 클라이언트의 조기 종료(스킵) 신호를 별도로 FastAPI에 전달할 계획이 있다면 메시지 포맷 협의가 필요하다.
3. **라이브 피드백(`feedback.data`)의 수치 포함 여부** — 현재는 Spring 문서 예시대로 문자열 배열만 담았는데, 실시간 진행바(WPM 수치 등)를 프론트가 매 구간 갱신해야 한다면 필드 확장이 필요하다(§3.4).
4. **`score`(0~100) 산출 기준** — Spring의 `PracticeSubStepResult.score` 저장을 위해 FastAPI가 매 subStep마다 단일 점수를 생성해야 하는데, 기존 실전 탭에는 이런 "단발 종합 점수" 개념이 없어 이번에 신규 정의가 필요하다(§5). subStep별 가중치/기준은 기획 확정 후 조정 예정.
