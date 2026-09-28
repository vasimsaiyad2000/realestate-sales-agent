from redis.asyncio import Redis
from app.core.config import settings


def create_redis_client() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)

redis_client = create_redis_client()