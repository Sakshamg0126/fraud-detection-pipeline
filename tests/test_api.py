from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.skipif(not Path("models/model.joblib").exists(), reason="model not trained")
def test_predict_rejects_missing_features():
    r = client.post("/predict", json={"features": {"V1": 0.5}})
    assert r.status_code == 422