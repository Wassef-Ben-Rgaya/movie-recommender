
import os
from pathlib import Path

import mlflow
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_PATH = MODEL_DIR / "svd_model.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
MLFLOW_EXPERIMENT_NAME = "movie-recommender"


def train_svd(ratings_df: pd.DataFrame) -> dict:
    """Train, evaluate and track an SVD recommendation model."""

    required_columns = {"user_id", "movie_id", "rating"}
    missing = required_columns - set(ratings_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ratings_df = ratings_df[
        ["user_id", "movie_id", "rating"]
    ].dropna().copy()

    if ratings_df.empty:
        raise ValueError("The ratings dataset is empty.")

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Prepare the MovieLens ratings dataset
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

    # Model hyperparameters
    params = {
        "algorithm": "SVD",
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
        "random_state": 42,
        "test_size": 0.20,
        "rating_scale": "1-5",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="svd-training") as run:
        mlflow.log_params(params)

        mlflow.set_tags({
            "model_family": "collaborative_filtering",
            "dataset": "MovieLens 100K",
        })

        model = SVD(
            n_factors=params["n_factors"],
            n_epochs=params["n_epochs"],
            lr_all=params["lr_all"],
            reg_all=params["reg_all"],
            random_state=params["random_state"],
        )

        print("Training SVD model...")
        model.fit(trainset)

        print("\nEvaluating model...")
        predictions = model.test(testset)

        rmse = float(accuracy.rmse(predictions, verbose=False))
        mae = float(accuracy.mae(predictions, verbose=False))

        # Save the model locally
        dump.dump(str(MODEL_PATH), algo=model)

        # Log metrics and the saved model file to MLflow
        metrics_to_log = {
            "rmse": rmse,
            "mae": mae,
            "training_ratings": trainset.n_ratings,
            "test_ratings": len(testset),
        }
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        mlflow.set_tag("model_format", "Surprise dump")

        print("\nSVD training completed!")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"Model saved locally to: {MODEL_PATH}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow server: {MLFLOW_TRACKING_URI}")

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "training_ratings": int(trainset.n_ratings),
            "test_ratings": len(testset),
            "model_path": str(MODEL_PATH),
        }

    return metrics
