from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.agroai.agents.yield_prediction import (
    YieldPredictionAgent,
    YieldPredictionRequest,
)
from src.agroai.data.yield_pipeline import (
    evaluate_regression,
    train_yield_models,
    validate_yield_data,
)
from main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "farmer_advisor_dataset.csv"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "yield_model.joblib"


def test_validate_yield_data():
    frame = pd.read_csv(DATASET_PATH)
    validated = validate_yield_data(frame)
    assert len(validated) == len(frame)
    assert "Crop_Type" in validated.columns
    assert "Crop_Yield_ton" in validated.columns


def test_train_yield_models_returns_real_metrics():
    frame = pd.read_csv(DATASET_PATH)
    artifact = train_yield_models(frame)
    test_metrics = artifact["test_metrics"]

    assert set(test_metrics) == {"mae", "rmse", "r2"}
    assert test_metrics["mae"] >= 0
    assert test_metrics["rmse"] >= 0
    assert -1 <= test_metrics["r2"] <= 1


def test_evaluate_regression():
    expected = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate_regression(expected, pd.Series([1.0, 2.0, 3.0]).to_numpy())
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_yield_prediction_endpoint():
    assert ARTIFACT_PATH.exists(), "Train the yield model before running API tests"
    client = TestClient(app)
    response = client.post(
        "/api/yield/predict",
        json={
            "crop": "Wheat",
            "location": "Karnataka",
            "soil_ph": 6.8,
            "soil_moisture": 30,
            "temperature_c": 24,
            "rainfall_mm": 180,
            "fertilizer_usage_kg": 120,
            "pesticide_usage_kg": 5,
            "sustainability_score": 70,
            "historical_yield": 6.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["crop"] == "Wheat"
    assert payload["unit"] == "tons"
    assert payload["predicted_yield"] >= 0
    assert payload["important_factors"]
    assert "lower" in payload["confidence_or_range"]
    assert "upper" in payload["confidence_or_range"]


def test_yield_agent_rejects_unknown_crop():
    if not ARTIFACT_PATH.exists():
        return
    agent = YieldPredictionAgent(ARTIFACT_PATH)
    try:
        agent.predict(YieldPredictionRequest(crop="Unknown-Crop"))
    except ValueError as exc:
        assert "Unsupported crop" in str(exc)
    else:
        raise AssertionError("Unknown crop should be rejected")
