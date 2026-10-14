
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, accuracy, dump
from surprise.model_selection import train_test_split
from sqlalchemy import text

from src.data.db import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "svd_model.pkl"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord l'entraînement avec : python -m src.models.train"
        )

    engine = get_engine()

    try:
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
        raise ValueError("Aucune évaluation trouvée dans la table ratings.")

    print(f"Évaluations chargées : {len(ratings_df):,}")

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]],
        reader,
    )

    _, testset = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
    )

    _, model = dump.load(str(MODEL_PATH))

    print("\nCalcul des prédictions...")
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    print("\nRésultats de l'évaluation SVD")
    print("-----------------------------")
    print(f"Nombre d'évaluations testées : {len(testset):,}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")


if __name__ == "__main__":
    main()
