from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_PATH = MODEL_DIR / "als_model.npz"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "movie-recommender"


def train_als(ratings_df: pd.DataFrame) -> dict:
    """Train an implicit ALS model and track the run in MLflow."""

    required_columns = {"user_id", "movie_id", "rating"}
    missing = required_columns - set(ratings_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ratings_df = ratings_df[
        ["user_id", "movie_id", "rating"]
    ].dropna().copy()

    if ratings_df.empty:
        raise ValueError("The ratings dataset is empty.")

    # Keep positive interactions only.
    positive = ratings_df[ratings_df["rating"] >= 4].copy()

    if positive.empty:
        raise ValueError("No positive interactions (rating >= 4).")

    # Map database IDs to consecutive matrix indices.
    user_ids = np.sort(ratings_df["user_id"].unique())
    movie_ids = np.sort(ratings_df["movie_id"].unique())

    user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    rows = positive["user_id"].map(user_to_idx).to_numpy()
    cols = positive["movie_id"].map(movie_to_idx).to_numpy()

    # Ratings of 4 or 5 become confidence weights.
    values = positive["rating"].astype(np.float32).to_numpy()

    user_items = coo_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    ).tocsr()

    factors = 64
    regularization = 0.1
    iterations = 20
    random_state = 42

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ALS"):
        mlflow.log_params(
            {
                "model_type": "AlternatingLeastSquares",
                "factors": factors,
                "regularization": regularization,
                "iterations": iterations,
                "random_state": random_state,
                "positive_rating_threshold": 4,
                "total_ratings": len(ratings_df),
            }
        )

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )

        print("Training ALS model...")
        model.fit(user_items, show_progress=True)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Preserve the existing local model format.
        np.savez_compressed(
            MODEL_PATH,
            user_factors=model.user_factors,
            item_factors=model.item_factors,
            user_ids=user_ids,
            movie_ids=movie_ids,
        )

        metrics = {
            "users": int(len(user_ids)),
            "movies": int(len(movie_ids)),
            "positive_interactions": int(len(positive)),
            "total_ratings": int(len(ratings_df)),
            "positive_interaction_ratio": float(
                len(positive) / len(ratings_df)
            ),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_files")

        mlflow.set_tags(
            {
                "model_type": "ALS",
                "feedback_type": "implicit",
                "positive_rating_threshold": "4",
            }
        )

    metrics["model_path"] = str(MODEL_PATH)

    print("\nALS training completed!")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics
