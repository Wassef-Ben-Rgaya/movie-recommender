
import csv
import json
import logging
import os
from pathlib import Path

# AWS Lambda : seul /tmp est accessible en écriture.
# Définir ces variables AVANT d'importer scikit-surprise.
os.environ["HOME"] = "/tmp"
os.environ["XDG_DATA_HOME"] = "/tmp"
os.environ["SURPRISE_DATA_FOLDER"] = "/tmp/surprise_data"

from surprise import dump
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get("S3_BUCKET")
if not S3_BUCKET:
    raise RuntimeError("La variable d'environnement S3_BUCKET est obligatoire.")

MODEL_PATH = Path(__file__).parent / "svd_model.pkl"

s3 = boto3.client("s3")

_model = None
_data = None


def load_model():
    """Charge le modèle SVD une seule fois par environnement Lambda."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Le modèle SVD est introuvable : {MODEL_PATH}"
            )

        _, _model = dump.load(str(MODEL_PATH))
        logger.info("SVD model loaded successfully")

    return _model


def load_data():
    """Charge les films et les notes depuis S3 et les conserve en mémoire."""
    global _data

    if _data is not None:
        return _data

    def read_csv(key):
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=key,
        )
        content = obj["Body"].read().decode("utf-8-sig")
        return list(csv.DictReader(content.splitlines()))

    movies = read_csv("data/movies.csv")
    ratings = read_csv("data/ratings.csv")

    titles = {}

    for row in movies:
        movie_id = row.get("movie_id") or row.get("movieId")
        title = row.get("title")

        if movie_id and title:
            titles[str(movie_id)] = title

    watched = {}

    for row in ratings:
        user_id = row.get("user_id") or row.get("userId")
        movie_id = row.get("movie_id") or row.get("movieId")

        if user_id and movie_id:
            watched.setdefault(str(user_id), set()).add(str(movie_id))

    users = set(watched.keys())

    _data = (titles, watched, users)

    logger.info(
        "Data loaded: %s movies, %s users",
        len(titles),
        len(users),
    )

    return _data


def response(status_code, body):
    """Construit une réponse compatible avec API Gateway HTTP API."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


def lambda_handler(event, context):
    """Retourne les meilleures recommandations SVD pour un utilisateur."""
    try:
        path = event.get("rawPath") or event.get("path") or ""
        parts = [part for part in path.split("/") if part]

        params = event.get("pathParameters") or {}

        user_id = (
            params.get("user_id")
            or params.get("userId")
            or (parts[-1] if parts else None)
        )

        if not user_id or user_id == "recommendations":
            return response(
                400,
                {
                    "error": (
                        "Provide a user ID in "
                        "/recommendations/{user_id}"
                    )
                },
            )

        query = event.get("queryStringParameters") or {}

        try:
            limit = int(query.get("n", query.get("limit", 10)))
        except (TypeError, ValueError):
            return response(400, {"error": "n must be an integer"})

        if not 1 <= limit <= 50:
            return response(
                400,
                {"error": "n must be between 1 and 50"},
            )

        model = load_model()
        titles, watched, users = load_data()

        user_id = str(user_id)

        if user_id not in users:
            return response(
                404,
                {"error": f"Unknown user_id: {user_id}"},
            )

        already_rated = watched[user_id]

        # Le modèle Surprise a été entraîné avec des identifiants entiers.
        raw_user_id = int(user_id) if user_id.isdigit() else user_id

        results = []

        for movie_id, title in titles.items():
            if movie_id in already_rated:
                continue

            raw_movie_id = (
                int(movie_id) if movie_id.isdigit() else movie_id
            )

            prediction = model.predict(raw_user_id, raw_movie_id)

            results.append(
                {
                    "movie_id": (
                        int(movie_id) if movie_id.isdigit() else movie_id
                    ),
                    "title": title,
                    "predicted_rating": round(
                        float(prediction.est),
                        2,
                    ),
                }
            )

        results.sort(
            key=lambda item: item["predicted_rating"],
            reverse=True,
        )

        recommendations = results[:limit]

        logger.info(
            "Generated %s SVD recommendations for user %s",
            len(recommendations),
            user_id,
        )

        return response(
            200,
            {
                "user_id": int(user_id) if user_id.isdigit() else user_id,
                "count": len(recommendations),
                "source": "SVD",
                "recommendations": recommendations,
            },
        )

    except Exception:
        logger.exception("SVD recommendation request failed")
        return response(
            500,
            {"error": "Internal server error"},
        )
