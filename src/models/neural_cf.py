
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_PATH = MODEL_DIR / "neural_cf.pt"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "movie-recommender"

DEVICE = torch.device("cpu")


class NeuralCF(nn.Module):
    """Neural collaborative filtering model for rating prediction."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        embedding_dim: int = 32,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)

    def forward(self, users, movies):
        user_vector = self.user_embedding(users)
        movie_vector = self.movie_embedding(movies)

        combined = torch.cat(
            [user_vector, movie_vector],
            dim=1,
        )

        return self.network(combined).squeeze(1)


def train_neural_cf(ratings_df: pd.DataFrame) -> dict:
    """Train NeuralCF, evaluate it, and track the run in MLflow."""

    required = {"user_id", "movie_id", "rating"}
    missing = required - set(ratings_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ratings_df = ratings_df[
        ["user_id", "movie_id", "rating"]
    ].dropna().copy()

    if ratings_df.empty:
        raise ValueError("The ratings dataset is empty.")

    # Reproducibility.
    random_state = 42
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    user_ids = np.sort(ratings_df["user_id"].unique())
    movie_ids = np.sort(ratings_df["movie_id"].unique())

    user_to_idx = {
        value: i for i, value in enumerate(user_ids)
    }
    movie_to_idx = {
        value: i for i, value in enumerate(movie_ids)
    }

    users = ratings_df["user_id"].map(user_to_idx).to_numpy()
    movies = ratings_df["movie_id"].map(movie_to_idx).to_numpy()
    ratings = ratings_df["rating"].to_numpy(dtype=np.float32)

    indices = np.arange(len(ratings))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
    )

    train_data = TensorDataset(
        torch.tensor(users[train_idx], dtype=torch.long),
        torch.tensor(movies[train_idx], dtype=torch.long),
        torch.tensor(ratings[train_idx], dtype=torch.float32),
    )

    test_users = torch.tensor(
        users[test_idx],
        dtype=torch.long,
    )
    test_movies = torch.tensor(
        movies[test_idx],
        dtype=torch.long,
    )
    test_ratings = ratings[test_idx]

    embedding_dim = 32
    batch_size = 1024
    learning_rate = 0.001
    weight_decay = 1e-5
    epochs = 20

    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Configure MLflow.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="NeuralCF"):
        mlflow.log_params(
            {
                "model_type": "NeuralCF",
                "framework": "PyTorch",
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "random_state": random_state,
                "n_users": len(user_ids),
                "n_movies": len(movie_ids),
                "total_ratings": len(ratings),
                "training_ratings": len(train_idx),
                "test_ratings": len(test_idx),
            }
        )

        model = NeuralCF(
            n_users=len(user_ids),
            n_movies=len(movie_ids),
            embedding_dim=embedding_dim,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()

        print("Training NeuralCF model...")

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch_users, batch_movies, batch_ratings in loader:
                batch_users = batch_users.to(DEVICE)
                batch_movies = batch_movies.to(DEVICE)
                batch_ratings = batch_ratings.to(DEVICE)

                optimizer.zero_grad()

                predictions = model(batch_users, batch_movies)
                loss = loss_fn(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_ratings)

            average_loss = total_loss / len(train_data)

            # Log training loss after every epoch.
            mlflow.log_metric(
                "train_mse",
                float(average_loss),
                step=epoch + 1,
            )

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- train MSE: {average_loss:.4f}"
                )

        # Evaluate on the held-out test set.
        model.eval()

        with torch.no_grad():
            predictions = model(
                test_users.to(DEVICE),
                test_movies.to(DEVICE),
            ).cpu().numpy()

        # Keep predicted ratings on the 1–5 scale.
        predictions = np.clip(predictions, 1.0, 5.0)

        rmse = float(
            np.sqrt(np.mean((predictions - test_ratings) ** 2))
        )
        mae = float(
            np.mean(np.abs(predictions - test_ratings))
        )

        # Preserve the existing local model format.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "user_ids": user_ids.tolist(),
                "movie_ids": movie_ids.tolist(),
                "embedding_dim": embedding_dim,
                "metrics": {
                    "rmse": rmse,
                    "mae": mae,
                },
            },
            MODEL_PATH,
        )

        # Log evaluation metrics to MLflow.
        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
            }
        )

        mlflow.set_tags(
            {
                "model_type": "NeuralCF",
                "framework": "PyTorch",
                "task": "rating_prediction",
            }
        )

        # Upload the saved model artifact to the MLflow run.
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model_files",
        )

        run_id = mlflow.active_run().info.run_id

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "training_ratings": len(train_idx),
        "test_ratings": len(test_idx),
        "model_path": str(MODEL_PATH),
        "mlflow_run_id": run_id,
    }

    print("\nNeuralCF training completed!")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(
        "\nMLflow run: "
        f"{MLFLOW_TRACKING_URI}/#/experiments/2/runs/{run_id}"
    )

    return metrics
