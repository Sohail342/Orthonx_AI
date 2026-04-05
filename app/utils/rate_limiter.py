"""Rate limiting utility for Chatbot endpoints using Redis."""

import time
from typing import Tuple

from redis import Redis

from app.core.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_redis_client() -> Redis:
    """Dependency for getting a Redis client."""
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)


def check_rate_limit(
    redis_client: Redis,
    ip_address: str,
    limit_type: str = "chat",
    limit: int = 10,
    period: int = 3600,
) -> Tuple[bool, int]:
    """Check if the given IP address has exceeded the rate limit.

    Args:
        redis_client: The Redis connection.
        ip_address: The user's IP.
        limit_type: The type of service being limited (e.g., 'chat', 'detect').
        limit: Max requests allowed (default 10).
        period: Time window in seconds (default 3600 i.e. 1 hour).

    Returns:
        Tuple[is_allowed, remaining_count]
    """
    # Create a unique key per IP, per type, and per hour block
    current_hour = int(time.time() / period)
    key = f"limit:{limit_type}:{ip_address}:{current_hour}"

    try:
        # Increment the count atomically
        count = redis_client.incr(key)

        # Set expiration if it's a new key
        if count == 1:
            redis_client.expire(key, period)

        if count > limit:
            return False, 0

        return True, limit - count

    except Exception as e:
        logger.error(f"Redis rate limiter error for IP {ip_address}: {e}")
        # Fail open in case of Redis failure (or change to fail closed based on policy)
        return True, 0
