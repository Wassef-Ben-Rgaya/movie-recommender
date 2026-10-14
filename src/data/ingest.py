
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-100k"


def load_users() -> pd.DataFrame:
    columns = [
        "user_id", "age", "gender", "occupation", "zip_code"
    ]

    users = pd.read_csv(
        DATA_DIR / "u.user",
        sep="|",
        header=None,
        names=columns,
        encoding="utf-8",
        dtype={"zip_code": str},
    )

    return users


def load_movies() -> pd.DataFrame:
    columns = [
        "movie_id", "title", "release_date",
        "video_release_date", "imdb_url",
    ] + [f"genre_{i}" for i in range(19)]

    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        header=None,
        names=columns,
        encoding="latin-1",
        dtype={"movie_id": "int64"},
    )

    genre_columns = [f"genre_{i}" for i in range(19)]
    genre_names = [
        "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western",
    ]

    movies["genres"] = movies[genre_columns].apply(
        lambda row: "|".join(
            name for name, active in zip(genre_names, row)
            if active == 1
        ),
        axis=1,
    )

    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        format="%d-%b-%Y",
        errors="coerce",
    ).dt.date

    return movies[[
        "movie_id", "title", "release_date", "genres"
    ]]


def load_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    ratings["rated_at"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
    ).dt.tz_localize(None)

    return ratings[[
        "user_id", "movie_id", "rating", "rated_at"
    ]]


def ingest_data() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    for filename in ["u.user", "u.item", "u.data"]:
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(
                f"Required dataset file not found: {filename}"
            )

    users = load_users()
    movies = load_movies()
    ratings = load_ratings()

    engine = get_engine()

    try:
        with engine.begin() as connection:
            # Reset these project tables for a clean initial import.
            connection.execute(text("""
                TRUNCATE TABLE
                    recommendations, ratings, movies, users
                RESTART IDENTITY CASCADE
            """))

            users.to_sql(
                "users", connection,
                if_exists="append", index=False,
                method="multi", chunksize=1000,
            )

            movies.to_sql(
                "movies", connection,
                if_exists="append", index=False,
                method="multi", chunksize=1000,
            )

            ratings.to_sql(
                "ratings", connection,
                if_exists="append", index=False,
                method="multi", chunksize=1000,
            )

        print("MovieLens ingestion completed!")
        print(f"Users: {len(users):,}")
        print(f"Movies: {len(movies):,}")
        print(f"Ratings: {len(ratings):,}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    ingest_data()
