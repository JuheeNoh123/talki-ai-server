import os, sys
from pathlib import Path

_TOPIC_MODEL_DIR = os.getenv(
    "TOPIC_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "Topic_model"),
)
if _TOPIC_MODEL_DIR not in sys.path:
    sys.path.insert(0, _TOPIC_MODEL_DIR)

from Topic_model.service_scorer import ServiceScorer

scorer = ServiceScorer()
