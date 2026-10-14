
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import dump


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_PATH = MODEL_DIR / "svd_model.pkl"


def train_svd(ratings_df: pd.DataFrame) -> dict:
    """Train an SVD recommendation model and evaluate its performance."""

    required_columns = {"user_id", "movie_id", "rating"}
    missing = required_columns - set(ratings_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ratings_df = ratings_df[
        ["user_id", "movie_id", "rating"]
    ].dropna().copy()

    if ratings_df.empty:
        raise ValueError("The ratings dataset is empty.")

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings_df, reader)

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

    print("Training SVD model...")
    model.fit(trainset)

    print("\nEvaluating model...")
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump.dump(str(MODEL_PATH), algo=model)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "training_ratings": int(trainset.n_ratings),
        "test_ratings": len(testset),
        "model_path": str(MODEL_PATH),
    }

    print("\nSVD training completed!")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    return metrics
