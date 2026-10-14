
from pathlib import Path

import numpy as np
from surprise import dump


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"


def test_svd_model_exists():
    model_path = MODEL_DIR / "svd_model.pkl"
    assert model_path.is_file()


def test_svd_prediction_is_valid():
    model_path = MODEL_DIR / "svd_model.pkl"
    _, model = dump.load(str(model_path))

    prediction = model.predict(uid=1, iid=475)

    assert np.isfinite(prediction.est)
    assert 1 <= prediction.est <= 5


def test_als_model_artifact():
    model_path = MODEL_DIR / "als_model.npz"
    assert model_path.is_file()

    with np.load(model_path) as data:
        assert data["user_factors"].shape[0] == 943
        assert data["item_factors"].shape[0] == 1682
        assert len(data["user_ids"]) == 943
        assert len(data["movie_ids"]) == 1682


def test_neural_cf_model_exists():
    model_path = MODEL_DIR / "neural_cf.pt"
    assert model_path.is_file()
    assert model_path.stat().st_size > 0
