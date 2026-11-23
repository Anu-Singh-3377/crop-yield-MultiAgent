import pandas as pd
import pytest

from src.agents.disease_agent import DiseasePredictionResponse
from src.agents.farmer_advisor import FarmerAdvisor, FarmerProfile
from src.agents.local_advisory_agent import LocalAdvisoryAgent
from src.agents.market_researcher import (
    MarketCropReport,
    MarketResearchOutput,
    MarketTrend,
)
from src.agents.weather_agent import WeatherResponse
from src.agroai.agents.yield_prediction import (
    ImportantFactor,
    YieldPredictionResponse,
)
from src.agroai.orchestration import AgroAIOrchestrator


class FakeFarmerAdvisor:
    def __init__(self):
        self.calls = 0

    def get_crop_recommendations(self, profile):
        self.calls += 1
        return [
            {
                "crop": "Wheat",
                "sustainability_score": 0.8,
                "soil_compatibility": 0.9,
                "water_requirement": 450,
                "estimated_cost": 100,
                "sustainability_metrics": {},
            }
        ]


class FakeMarketResearcher:
    def __init__(self):
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return MarketResearchOutput(
            region=request.region,
            recommendations=[
                MarketCropReport(
                    crop=request.crops[0],
                    market_trend=MarketTrend(
                        crop=request.crops[0],
                        current_price=300,
                        predicted_price=315,
                        demand_level="High",
                        price_trend="Increasing",
                        confidence_score=0.7,
                    ),
                    profitability={"estimated_profit": 1000},
                    recommendation_score=0.75,
                )
            ],
        )


class FakeWeatherAgent:
    def __init__(self, fails=False):
        self.calls = 0
        self.fails = fails

    async def get_weather(self, **kwargs):
        self.calls += 1
        if self.fails:
            raise RuntimeError("weather service unavailable")
        return WeatherResponse(
            location=kwargs.get("location") or "12.0, 77.0",
            temperature=25,
            humidity=60,
            rainfall=2,
        )


class FakeDiseaseAgent:
    def __init__(self, fails=False):
        self.calls = 0
        self.fails = fails

    def predict(self, image, crop_hint=None):
        self.calls += 1
        if self.fails:
            raise RuntimeError("model artifact unavailable")
        return DiseasePredictionResponse(
            crop=crop_hint or "Wheat",
            disease="healthy",
            confidence=0.9,
            risk_level="low",
        )


class FakeYieldAgent:
    def __init__(self):
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        return YieldPredictionResponse(
            crop=request.crop,
            location=request.location,
            predicted_yield=4.2,
            confidence_or_range={"lower": 3.8, "upper": 4.6},
            important_factors=[ImportantFactor(factor="Rainfall_mm", impact=0.2)],
            model_version="test",
            historical_yield=request.historical_yield,
        )


class MustNotRun:
    def __getattr__(self, name):
        raise AssertionError(f"Unexpected agent invocation: {name}")


class CountingAdvisory(LocalAdvisoryAgent):
    def __init__(self):
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return super().run(request)


@pytest.mark.asyncio
async def test_yield_only_does_not_call_unneeded_agents():
    yield_agent = FakeYieldAgent()
    orchestrator = AgroAIOrchestrator(
        farmer_advisor=MustNotRun(),
        market_researcher=MustNotRun(),
        weather_agent=MustNotRun(),
        disease_agent=MustNotRun(),
        yield_prediction_agent=yield_agent,
    )

    result = await orchestrator.ainvoke({"intent": "yield_only", "crop": "Wheat"})

    assert result.yield_prediction.predicted_yield == 4.2
    assert yield_agent.calls == 1
    assert set(result.executed_agents) == {"yield_prediction", "local_advisory"}
    assert "disease" in result.skipped_agents


@pytest.mark.asyncio
async def test_disease_failure_does_not_block_weather_or_advisory():
    disease_agent = FakeDiseaseAgent(fails=True)
    weather_agent = FakeWeatherAgent()
    advisory = CountingAdvisory()
    orchestrator = AgroAIOrchestrator(
        weather_agent=weather_agent,
        disease_agent=disease_agent,
        local_advisory_agent=advisory,
    )

    result = await orchestrator.ainvoke(
        {
            "intent": "disease_detection",
            "image": b"sample image",
            "crop": "Wheat",
            "location": "Bengaluru",
        }
    )

    assert disease_agent.calls == 1
    assert weather_agent.calls == 1
    assert result.weather.location == "Bengaluru"
    assert result.disease is None
    assert result.final_advisory is not None
    assert advisory.calls == 1
    assert "disease" in result.executed_agents
    assert any(error.agent == "disease" for error in result.agent_errors)


@pytest.mark.asyncio
async def test_full_assessment_calls_all_available_agents():
    farmer = FakeFarmerAdvisor()
    market = FakeMarketResearcher()
    weather = FakeWeatherAgent()
    disease = FakeDiseaseAgent()
    yield_agent = FakeYieldAgent()
    orchestrator = AgroAIOrchestrator(
        farmer_advisor=farmer,
        market_researcher=market,
        weather_agent=weather,
        disease_agent=disease,
        yield_prediction_agent=yield_agent,
    )
    profile = FarmerProfile(
        name="Asha",
        location="Bengaluru",
        farm_size=2,
        soil_type="loamy",
        water_availability="moderate",
        preferred_crops=["Wheat"],
        budget=10000,
    )

    result = await orchestrator.ainvoke(
        {
            "intent": "farm_assessment",
            "farmer_profile": profile,
            "crop": "Wheat",
            "market_crops": ["Wheat"],
            "image": b"sample image",
        }
    )

    assert farmer.calls == market.calls == weather.calls == disease.calls == 1
    assert yield_agent.calls == 1
    assert set(result.executed_agents) == {
        "farmer_advisor",
        "market_researcher",
        "weather",
        "disease",
        "yield_prediction",
        "local_advisory",
    }
    assert result.final_advisory is not None
    assert result.market_analysis.region == "Bengaluru"
    assert result.sustainability_assessment[0].crop == "Wheat"


@pytest.mark.asyncio
async def test_real_farmer_advisor_metrics_are_normalized_for_advisory():
    orchestrator = AgroAIOrchestrator(
        farmer_advisor=FarmerAdvisor(pd.DataFrame()),
        market_researcher=FakeMarketResearcher(),
        weather_agent=FakeWeatherAgent(),
        yield_prediction_agent=FakeYieldAgent(),
    )
    profile = FarmerProfile(
        name="Asha",
        location="Bengaluru",
        farm_size=2,
        soil_type="loamy",
        water_availability="moderate",
        preferred_crops=["Wheat"],
        budget=10000,
    )

    result = await orchestrator.ainvoke(
        {
            "intent": "farm_assessment",
            "farmer_profile": profile,
            "crop": "Wheat",
            "market_crops": ["Wheat"],
        }
    )

    assert result.agent_errors == []
    assert result.farmer_recommendations[0]["sustainability_metrics"]["soil_health"] > 0
    assert result.sustainability_assessment[0].crop == "Wheat"
    assert result.final_advisory is not None


@pytest.mark.asyncio
async def test_invalid_request_does_not_run_agents():
    weather_agent = FakeWeatherAgent()
    orchestrator = AgroAIOrchestrator(weather_agent=weather_agent)

    result = await orchestrator.ainvoke({"intent": "weather"})

    assert weather_agent.calls == 0
    assert result.agent_errors[0].agent == "input"
    assert result.executed_agents == []
    assert result.final_advisory is None
