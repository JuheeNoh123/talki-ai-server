# app/schemas/analyze_schema.py
from pydantic import BaseModel
from typing import Optional

class AnalyzeFromS3Request(BaseModel):
    video_url: str
    s3_key: str
    presentation_type: str

    topic_summary: str
    topic_desc: str
    topic_tags: list

    presentation_id: Optional[str] = None  # WebSocket 세션 ID (돌발 질문 결과 조회용)

