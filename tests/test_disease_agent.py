from io import BytesIO

import pytest

from src.agents.disease_agent import DiseaseAgent


def test_missing_model_has_explicit_error(tmp_path):
    agent = DiseaseAgent(tmp_path / "missing-model.pt")
    with pytest.raises(FileNotFoundError, match="Disease model artifact not found"):
        agent.predict(BytesIO(b"not-an-image"), crop_hint="Tomato")


def test_class_name_is_split_into_crop_and_disease():
    agent = DiseaseAgent()
    crop, disease = agent._split_crop_and_disease("Tomato___Early_blight")
    assert crop == "Tomato"
    assert disease == "Early blight"


def test_risk_levels():
    agent = DiseaseAgent()
    assert agent._risk_level("Healthy", 0.99) == "low"
    assert agent._risk_level("Early_blight", 0.90) == "high"
    assert agent._risk_level("Early_blight", 0.70) == "medium"
    assert agent._risk_level("Early_blight", 0.40) == "low"
