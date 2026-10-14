
from sqlalchemy import text
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from src.data.db import get_engine


def main():
    engine = get_engine()

    try:
        print("Loading ratings from PostgreSQL...")

        with engine.connect() as connection:
            ratings_df = pd.read_sql(
                text("""
                    SELECT user_id, movie_id, rating
                    FROM ratings
                    ORDER BY user_id, movie_id
                """),
                connection,
            )
    finally:
        engine.dispose()

    if ratings_df.empty:
        raise ValueError("The ratings dataset is empty.")

    required = {"user_id", "movie_id", "rating"}
    if not required.issubset(ratings_df.columns):
        raise ValueError("Required rating columns are missing.")

    ratings_df = ratings_df[
        ["user_id", "movie_id", "rating"]
    ].dropna()

    print(f"Ratings loaded: {len(ratings_df):,}")

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]],
        reader,
    )

    trainset, testset = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )

    print("Training a fresh SVD model...")
    model.fit(trainset)

    print("Evaluating on held-out test ratings...")
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    print("\nSVD evaluation results")
    print("----------------------")
    print(f"Training ratings: {trainset.n_ratings:,}")
    print(f"Test ratings:     {len(testset):,}")
    print(f"RMSE:             {rmse:.4f}")
    print(f"MAE:              {mae:.4f}")


if __name__ == "__main__":
    main()
