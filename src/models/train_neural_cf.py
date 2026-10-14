
import pandas as pd
from sqlalchemy import text

from src.data.db import get_engine
from src.models.neural_cf import train_neural_cf


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

        print(f"Loaded {len(ratings_df):,} ratings.")

        metrics = train_neural_cf(ratings_df)

        print("\nFinal metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
