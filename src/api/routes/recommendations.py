
from fastapi import APIRouter, HTTPException, Query

from src.cache.redis_client import (
    get_cached_recommendations,
    set_cached_recommendations,
)
from src.models.recommender import recommend_movies

router = APIRouter(tags=["Recommendations"])


@router.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    """Return personalized recommendations, using Redis when available."""

    try:
        # 1. Try Redis first.
        cached = get_cached_recommendations(user_id=user_id, n=n)

        if cached is not None:
            return {
                "user_id": user_id,
                "count": len(cached),
                "recommendations": cached,
                "source": "redis",
            }

        # 2. Cache miss: use the existing SVD recommendation engine.
        results = recommend_movies(user_id=user_id, n=n)
        recommendations = results.to_dict(orient="records")

        # 3. Cache the result. A Redis failure must not break the API.
        set_cached_recommendations(
            user_id=user_id,
            n=n,
            recommendations=recommendations,
        )

        return {
            "user_id": user_id,
            "count": len(recommendations),
            "recommendations": recommendations,
            "source": "model",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
