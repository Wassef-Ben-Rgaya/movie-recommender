
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from airflow.sdk import dag, task


@dag(
    dag_id="movie_recommender_test",
    description="Pipeline complet d'entraînement Movie Recommender",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["movie-recommender", "training"],
)
def movie_recommender_test():

    @task
    def check_database():
        from sqlalchemy import text
        from src.data.db import get_engine

        engine = get_engine()
        try:
            with engine.connect() as connection:
                count = connection.execute(
                    text("SELECT COUNT(*) FROM ratings")
                ).scalar_one()

            print(f"Nombre d'évaluations disponibles : {count}")

            if count == 0:
                raise ValueError("La table ratings est vide.")

            return count
        finally:
            engine.dispose()

    @task
    def train_svd_model():
        subprocess.run(
            [sys.executable, "-m", "src.models.train"],
            check=True,
        )

    @task
    def train_als_model():
        subprocess.run(
            [sys.executable, "-m", "src.models.train_als"],
            check=True,
        )

    @task
    def train_neural_cf_model():
        subprocess.run(
            [sys.executable, "-m", "src.models.train_neural_cf"],
            check=True,
        )

    @task
    def verify_models():
        model_dir = Path(
            "/opt/airflow/project/data/processed/models"
        )

        expected_models = [
            "svd_model.pkl",
            "als_model.npz",
            "neural_cf.pt",
        ]

        missing = [
            name
            for name in expected_models
            if not (model_dir / name).is_file()
            or (model_dir / name).stat().st_size == 0
        ]

        if missing:
            raise FileNotFoundError(
                f"Modèles absents ou vides : {missing}"
            )

        for name in expected_models:
            path = model_dir / name
            print(f"Modèle vérifié : {path} ({path.stat().st_size} octets)")

        print("Les trois modèles sont présents.")

    database_check = check_database()
    svd = train_svd_model()
    als = train_als_model()
    neural_cf = train_neural_cf_model()
    verification = verify_models()

    database_check >> svd >> als >> neural_cf >> verification


movie_recommender_test()

