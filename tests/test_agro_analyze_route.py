import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agroai.orchestration import AgroAIResponse
from src.api import routes


class FakeOrchestrator:
    def __init__(self):
        self.payload = None

    async def ainvoke(self, payload):
        self.payload = payload
        return AgroAIResponse(
            intent=payload["intent"],
            crop=payload.get("crop"),
            executed_agents=["disease", "local_advisory"],
        )


def build_client(monkeypatch):
    orchestrator = FakeOrchestrator()
    monkeypatch.setattr(routes, "agroai_orchestrator", orchestrator)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app), orchestrator


def test_agro_analyze_route_decodes_base64_image(monkeypatch):
    client, orchestrator = build_client(monkeypatch)
    image = b"test image bytes"

    response = client.post(
        "/api/agro/analyze",
        json={
            "intent": "disease_detection",
            "crop": "Wheat",
            "image_base64": base64.b64encode(image).decode("ascii"),
        },
    )

    assert response.status_code == 200
    assert response.json()["crop"] == "Wheat"
    assert orchestrator.payload["image"] == image


def test_agro_analyze_rejects_invalid_base64(monkeypatch):
    client, orchestrator = build_client(monkeypatch)

    response = client.post(
        "/api/agro/analyze",
        json={
            "intent": "disease_detection",
            "image_base64": "not valid base64!",
        },
    )

    assert response.status_code == 422
    assert "valid Base64" in response.json()["detail"]
    assert orchestrator.payload is None


def test_agro_analyze_preserves_typed_farmer_profile(monkeypatch):
    client, orchestrator = build_client(monkeypatch)

    response = client.post(
        "/api/agro/analyze",
        json={
            "intent": "farm_assessment",
            "farmer_profile": {
                "name": "Asha",
                "location": "Bengaluru",
                "farm_size": 2,
                "soil_type": "loamy",
                "water_availability": "moderate",
                "preferred_crops": ["Wheat"],
                "budget": 10000,
            },
        },
    )

    assert response.status_code == 200
    assert orchestrator.payload["farmer_profile"]["name"] == "Asha"
