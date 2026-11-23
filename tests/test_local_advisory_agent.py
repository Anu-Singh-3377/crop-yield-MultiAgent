from src.agents.disease_agent import DiseasePredictionResponse
from src.agents.farmer_advisor import FarmerProfile
from src.agents.local_advisory_agent import (
    LocalAdvisoryAgent,
    LocalAdvisoryInput,
)
from src.agents.market_researcher import (
    MarketCropReport,
    MarketResearchOutput,
    MarketTrend,
)
from src.agents.weather_agent import (
    AgriculturalRisk,
    WeatherForecastDay,
    WeatherResponse,
)
from src.agroai.agents.yield_prediction import (
    ImportantFactor,
    YieldPredictionResponse,
)


def test_agent_consumes_all_upstream_outputs_without_recalculating_them():
    farmer_profile = FarmerProfile(
        name="Asha",
        location="Karnataka",
        farm_size=2.0,
        soil_type="loamy",
        water_availability="low",
        preferred_crops=["Wheat"],
        budget=10000,
    )
    market = MarketResearchOutput(
        region="Karnataka",
        recommendations=[
            MarketCropReport(
                crop="Wheat",
                market_trend=MarketTrend(
                    crop="Wheat",
                    current_price=300,
                    predicted_price=315,
                    demand_level="Medium",
                    price_trend="Increasing",
                    confidence_score=0.8,
                ),
                profitability={
                    "estimated_yield": 6,
                    "potential_revenue": 1800,
                    "production_cost": 720,
                    "profit_margin": 0.6,
                    "roi": 1.5,
                },
                recommendation_score=0.9,
            )
        ],
    )
    weather = WeatherResponse(
        location="Karnataka",
        temperature=39,
        humidity=88,
        rainfall=25,
        forecast=[
            WeatherForecastDay(
                date="2026-11-23",
                temperature_max=39,
                temperature_min=25,
                rainfall=25,
            )
        ],
        agricultural_risks=[
            AgriculturalRisk(
                type="heat_stress",
                severity="high",
                description="Heat stress risk reported by weather agent.",
            ),
            AgriculturalRisk(
                type="heavy_rain",
                severity="high",
                description="Heavy rain risk reported by weather agent.",
            ),
        ],
    )
    disease = DiseasePredictionResponse(
        crop="Wheat",
        disease="Leaf rust",
        confidence=0.58,
        risk_level="low",
    )
    yield_prediction = YieldPredictionResponse(
        crop="Wheat",
        predicted_yield=4.2,
        unit="tons",
        confidence_or_range={
            "type": "empirical_95_percent_residual_range",
            "lower": 2.0,
            "upper": 6.4,
        },
        important_factors=[
            ImportantFactor(factor="Rainfall_mm", impact=0.12),
        ],
        model_version="xgboost-v1",
    )

    output = LocalAdvisoryAgent().run(LocalAdvisoryInput(
        farmer_profile=farmer_profile,
        farmer_recommendations=[
            {
                "crop": "Wheat",
                "sustainability_score": 0.62,
                "soil_compatibility": 0.75,
                "water_requirement": 900,
                "estimated_cost": 450,
                "sustainability_metrics": {
                    "water_efficiency": 0.4,
                    "soil_health": 0.45,
                },
            }
        ],
        market_research=market,
        weather=weather,
        disease=disease,
        yield_prediction=yield_prediction,
    ))

    assert output.crop_assessment.candidates[0].crop == "Wheat"
    assert output.disease_risk.disease == "Leaf rust"
    assert output.disease_risk.confidence == 0.58
    assert output.weather_risk.temperature == 39
    assert output.weather_risk.agricultural_risks[0].type == "heat_stress"
    assert output.yield_assessment.predicted_yield == 4.2
    assert output.yield_assessment.confidence_or_range == yield_prediction.confidence_or_range
    assert output.market_assessment.recommendations[0].recommendation_score == 0.9
    assert any("not a guaranteed" in warning for warning in output.warnings)
    assert any("confirm the diagnosis" in warning for warning in output.warnings)
    assert any("irrigation" in action.action for action in output.recommended_actions)
    assert any("water efficiency" in action.rationale for action in output.sustainability_actions)


def test_missing_agent_outputs_are_reported_not_fabricated():
    output = LocalAdvisoryAgent().run(LocalAdvisoryInput())

    assert output.crop_assessment.status == "not_available"
    assert output.disease_risk.status == "not_available"
    assert output.weather_risk.status == "not_available"
    assert output.yield_assessment.status == "not_available"
    assert output.market_assessment.status == "not_available"
    assert output.recommended_actions == []
    assert len(output.warnings) >= 6


def test_healthy_disease_result_does_not_create_treatment_action():
    output = LocalAdvisoryAgent().run(LocalAdvisoryInput(
        disease=DiseasePredictionResponse(
            crop="Tomato",
            disease="healthy",
            confidence=0.99,
            risk_level="low",
        ),
    ))

    assert output.disease_risk.disease == "healthy"
    assert not any(
        "treatment" in action.action.lower()
        for action in output.recommended_actions
    )
