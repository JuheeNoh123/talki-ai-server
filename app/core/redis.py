import redis
import os

# REDIS_HOST = os.getenv("REDIS_HOST", "43.201.182.246")
# REDIS_PASS = os.getenv("REDIS_PASS", "talki1234")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_DB = int(os.getenv("REDIS_DB", 0))

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    password="talki1234",
    #db=REDIS_DB,
    decode_responses=True  # JSON 문자열 바로 쓰기 위함
)
