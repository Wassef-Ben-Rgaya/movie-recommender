
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from surprise import dump

from src.data.db import get_engine


# Racine du projet : movie-recommender/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chemin portable du modèle SVD
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "svd_model.pkl"
)


def recommend_movies(user_id: int, n: int = 10) -> pd.DataFrame:
    """Recommend unseen movies to an existing MovieLens user."""

    if n < 1:
        raise ValueError("n must be greater than or equal to 1.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SVD model not found: {MODEL_PATH}"
        )

    _, model = dump.load(str(MODEL_PATH))
    engine = get_engine()

    try:
        with engine.connect() as connection:
            user_result = connection.execute(
                text("SELECT user_id FROM users WHERE user_id = :uid"),
                {"uid": user_id},
            ).first()

            if user_result is None:
                raise ValueError(f"User {user_id} does not exist.")

            movies = pd.read_sql(
                text("SELECT movie_id, title FROM movies"),
                connection,
            )

            watched = pd.read_sql(
                text("""
                    SELECT movie_id
                    FROM ratings
                    WHERE user_id = :uid
                """),
                connection,
                params={"uid": user_id},
            )

        watched_ids = set(watched["movie_id"].tolist())

        candidates = movies[
            ~movies["movie_id"].isin(watched_ids)
        ].copy()

        if candidates.empty:
            return pd.DataFrame(
                columns=["movie_id", "title", "predicted_rating"]
            )

        candidates["predicted_rating"] = candidates["movie_id"].apply(
            lambda movie_id: model.predict(
                uid=user_id,
                iid=int(movie_id),
            ).est
        )

        recommendations = candidates.sort_values(
            "predicted_rating",
            ascending=False,
        ).head(n)

        recommendations["predicted_rating"] = (
            recommendations["predicted_rating"].round(2)
        )

        return recommendations[
            ["movie_id", "title", "predicted_rating"]
        ].reset_index(drop=True)

    finally:
        engine.dispose()


if __name__ == "__main__":
    selected_user = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    results = recommend_movies(selected_user)

    print(f"\nTop recommendations for user {selected_user}:\n")

    if results.empty:
        print("No recommendations available.")
    else:
        print(results.to_string(index=False))
