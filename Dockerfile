# CUDA 12.1 + Python 3.11 베이스 이미지
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python3.11 -m pip install --upgrade pip

# torch CUDA 12.1 빌드 먼저 설치 (requirements.txt 변경과 무관하게 캐시 유지)
RUN pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --retries 5 --timeout 600

# 작업 디렉토리
WORKDIR /app

# requirements 복사 후 나머지 패키지 설치
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# 소스코드 복사 (Topic_model 포함 - .dockerignore에서 제외됨)
COPY . .

# 환경변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]