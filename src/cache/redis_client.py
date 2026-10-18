
import json
import os

import redis


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL_SECONDS = int(
    os.getenv("RECOMMENDATIONS_CACHE_TTL", "300")
)

_client = None


def get_redis_client():
    """Return a reusable Redis client."""
    global _client

    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    return _client


def get_cached_recommendations(user_id: int, n: int):
    """Return cached recommendations, or None on a cache miss."""
    key = f"recommendations:{user_id}:{n}"

    try:
        client = get_redis_client()
        value = client.get(key)

        if value is None:
            return None

        result = json.loads(value)

        if not isinstance(result, list):
            client.delete(key)
            return None

        return result

    except (
        redis.RedisError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Redis cache read error: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def set_cached_recommendations(
    user_id: int,
    n: int,
    recommendations: list[dict],
) -> bool:
    """Save recommendations in Redis with a TTL."""
    key = f"recommendations:{user_id}:{n}"

    try:
        payload = json.dumps(recommendations)
        result = get_redis_client().set(
            key,
            payload,
            ex=CACHE_TTL_SECONDS,
        )

        if result:
            print(f"Redis cache saved: {key}")

        return bool(result)

    except (
        redis.RedisError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Redis cache write error: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def invalidate_user_recommendations(user_id: int) -> int:
    """Delete all cached recommendation lists for a user.

    Return the number of deleted keys.
    Propagate Redis errors so the caller can handle retries.
    """
    pattern = f"recommendations:{user_id}:*"
    client = get_redis_client()
    deleted = 0

    try:
        for key in client.scan_iter(match=pattern, count=100):
            deleted += int(client.delete(key))

        return deleted

    except redis.RedisError as exc:
        print(
            f"Redis cache invalidation error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise


def check_redis_connection() -> bool:
    """Check whether Redis is reachable."""
    try:
        return bool(get_redis_client().ping())

    except redis.RedisError as exc:
        print(f"Redis connection error: {exc}")
        return False
