
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Movie Recommender API"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("src.api.routes.recommendations.recommend_movies")
def test_recommendations_success(mock_recommend):
    mock_recommend.return_value = pd.DataFrame(
        [
            {
                "movie_id": 475,
                "title": "Trainspotting (1996)",
                "predicted_rating": 4.76,
            },
            {
                "movie_id": 483,
                "title": "Casablanca (1942)",
                "predicted_rating": 4.74,
            },
        ]
    )

    response = client.get("/recommendations/1?n=2")

    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == 1
    assert data["count"] == 2
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["title"] == "Trainspotting (1996)"


@patch("src.api.routes.recommendations.recommend_movies")
def test_recommendations_unknown_user(mock_recommend):
    mock_recommend.side_effect = ValueError("User 9999 does not exist.")

    response = client.get("/recommendations/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "User 9999 does not exist."


def test_recommendations_invalid_limit():
    response = client.get("/recommendations/1?n=0")

    assert response.status_code == 422


def test_recommendations_limit_above_maximum():
    response = client.get("/recommendations/1?n=51")

    assert response.status_code == 422
