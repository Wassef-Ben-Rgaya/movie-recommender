
from fastapi import APIRouter, HTTPException, Query

from src.models.recommender import recommend_movies

router = APIRouter(tags=["Recommendations"])


@router.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    """Return personalized movie recommendations for a user."""

    try:
        results = recommend_movies(user_id=user_id, n=n)

        return {
            "user_id": user_id,
            "count": len(results),
            "recommendations": results.to_dict(orient="records"),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
