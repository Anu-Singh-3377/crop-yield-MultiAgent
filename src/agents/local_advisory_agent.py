from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from .disease_agent import DiseasePredictionResponse
from .farmer_advisor import FarmerProfile
from .market_researcher import MarketCropReport, MarketResearchOutput
from .weather_agent import AgriculturalRisk, WeatherForecastDay, WeatherResponse
from ..agroai.agents.yield_prediction import (
    ImportantFactor,
    YieldPredictionResponse,
)


class CropCandidateAssessment(BaseModel):
    crop: str
    sustainability_score: Optional[float] = None
    soil_compatibility: Optional[float] = None
    water_requirement: Optional[float] = None
    estimated_cost: Optional[float] = None
    sustainability_metrics: dict[str, Any] = Field(default_factory=dict)


class CropAssessment(BaseModel):
    status: Literal["available", "not_available"]
    location: Optional[str] = None
    soil_type: Optional[str] = None
    water_availability: Optional[str] = None
    preferred_crops: List[str] = Field(default_factory=list)
    candidates: List[CropCandidateAssessment] = Field(default_factory=list)
    source: str = "Farmer Advisor"


class DiseaseRiskAssessment(BaseModel):
    status: Literal["available", "not_available"]
    crop: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    source: str = "Crop Disease Detection Agent"
    interpretation: str


class WeatherRiskAssessment(BaseModel):
    status: Literal["available", "not_available"]
    location: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    forecast: List[WeatherForecastDay] = Field(default_factory=list)
    agricultural_risks: List[AgriculturalRisk] = Field(default_factory=list)
    source: str = "Weather Agent"


class YieldAssessment(BaseModel):
    status: Literal["available", "not_available"]
    crop: Optional[str] = None
    predicted_yield: Optional[float] = None
    unit: Optional[str] = None
    confidence_or_range: Optional[dict[str, Any]] = None
    important_factors: List[ImportantFactor] = Field(default_factory=list)
    model_version: Optional[str] = None
    interpretation: str
    source: str = "Yield Prediction Agent"


class MarketAssessment(BaseModel):
    status: Literal["available", "not_available"]
    region: Optional[str] = None
    recommendations: List[MarketCropReport] = Field(default_factory=list)
    interpretation: str
    source: str = "Market Researcher Agent"


class AdvisoryAction(BaseModel):
    action: str
    rationale: str
    priority: Literal["low", "medium", "high"]
    evidence: List[str] = Field(default_factory=list)


class LocalAdvisoryInput(BaseModel):
    farmer_profile: Optional[FarmerProfile] = None
    farmer_recommendations: List[dict[str, Any]] = Field(default_factory=list)
    market_research: Optional[MarketResearchOutput] = None
    weather: Optional[WeatherResponse] = None
    disease: Optional[DiseasePredictionResponse] = None
    yield_prediction: Optional[YieldPredictionResponse] = None


class LocalAdvisoryOutput(BaseModel):
    crop_assessment: CropAssessment
    disease_risk: DiseaseRiskAssessment
    weather_risk: WeatherRiskAssessment
    yield_assessment: YieldAssessment
    market_assessment: MarketAssessment
    sustainability_actions: List[AdvisoryAction] = Field(default_factory=list)
    recommended_actions: List[AdvisoryAction] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LocalAdvisoryAgent:
    """Creates conservative advice from supplied agent outputs without an LLM."""

    def run(self, request: LocalAdvisoryInput) -> LocalAdvisoryOutput:
        warnings: List[str] = []
        crop_assessment = self._crop_assessment(request, warnings)
        disease_risk = self._disease_assessment(request.disease, warnings)
        weather_risk = self._weather_assessment(request.weather, warnings)
        yield_assessment = self._yield_assessment(
            request.yield_prediction,
            warnings,
        )
        market_assessment = self._market_assessment(
            request.market_research,
            warnings,
        )

        sustainability_actions = self._sustainability_actions(
            request.farmer_recommendations,
            request.weather,
        )
        recommended_actions = self._recommended_actions(
            request.disease,
            request.weather,
            request.market_research,
            request.yield_prediction,
        )
        if not any((
            request.farmer_profile,
            request.farmer_recommendations,
            request.market_research,
            request.weather,
            request.disease,
            request.yield_prediction,
        )):
            warnings.append(
                "No upstream agent outputs were provided; no farm-specific advice was generated."
            )

        return LocalAdvisoryOutput(
            crop_assessment=crop_assessment,
            disease_risk=disease_risk,
            weather_risk=weather_risk,
            yield_assessment=yield_assessment,
            market_assessment=market_assessment,
            sustainability_actions=sustainability_actions,
            recommended_actions=recommended_actions,
            warnings=warnings,
        )

    def _crop_assessment(
        self,
        request: LocalAdvisoryInput,
        warnings: List[str],
    ) -> CropAssessment:
        profile = request.farmer_profile
        candidates = [
            CropCandidateAssessment.model_validate(item)
            for item in request.farmer_recommendations
        ]
        if profile is None and not candidates:
            warnings.append("Farmer Advisor output is unavailable.")
        return CropAssessment(
            status="available" if profile is not None or candidates else "not_available",
            location=profile.location if profile else None,
            soil_type=profile.soil_type if profile else None,
            water_availability=profile.water_availability if profile else None,
            preferred_crops=(profile.preferred_crops or []) if profile else [],
            candidates=candidates,
        )

    def _disease_assessment(
        self,
        disease: Optional[DiseasePredictionResponse],
        warnings: List[str],
    ) -> DiseaseRiskAssessment:
        if disease is None:
            warnings.append("Crop Disease Detection Agent output is unavailable.")
            return DiseaseRiskAssessment(
                status="not_available",
                interpretation="No image-based disease result was supplied.",
            )

        healthy = disease.disease.strip().lower() in {"healthy", "normal"}
        interpretation = (
            "The supplied classifier result is labeled healthy; continue routine scouting."
            if healthy
            else (
                "This is a model classification, not a confirmed diagnosis. "
                "Inspect the plant and seek qualified local confirmation before treatment."
            )
        )
        if not healthy:
            warnings.append(
                "Disease output is a model prediction; confirm the diagnosis before choosing treatment."
            )
        if disease.confidence < 0.6:
            warnings.append(
                "Disease model confidence is low; do not act on this classification alone."
            )
        return DiseaseRiskAssessment(
            status="available",
            crop=disease.crop,
            disease=disease.disease,
            confidence=disease.confidence,
            risk_level=disease.risk_level,
            interpretation=interpretation,
        )

    def _weather_assessment(
        self,
        weather: Optional[WeatherResponse],
        warnings: List[str],
    ) -> WeatherRiskAssessment:
        if weather is None:
            warnings.append("Weather Agent output is unavailable.")
            return WeatherRiskAssessment(
                status="not_available",
            )
        if weather.provider_status != "ok":
            warnings.append(
                "Weather provider status is not OK; verify weather data before using it."
            )
        return WeatherRiskAssessment(
            status="available",
            location=weather.location,
            temperature=weather.temperature,
            humidity=weather.humidity,
            rainfall=weather.rainfall,
            forecast=weather.forecast,
            agricultural_risks=weather.agricultural_risks,
        )

    def _yield_assessment(
        self,
        prediction: Optional[YieldPredictionResponse],
        warnings: List[str],
    ) -> YieldAssessment:
        if prediction is None:
            warnings.append("Yield Prediction Agent output is unavailable.")
            return YieldAssessment(
                status="not_available",
                interpretation="No model prediction was supplied.",
            )
        warnings.append(
            "Yield is a model prediction with uncertainty, not a guaranteed harvest outcome."
        )
        return YieldAssessment(
            status="available",
            crop=prediction.crop,
            predicted_yield=prediction.predicted_yield,
            unit=prediction.unit,
            confidence_or_range=prediction.confidence_or_range,
            important_factors=prediction.important_factors,
            model_version=prediction.model_version,
            interpretation=(
                "Use the supplied model estimate and range for planning; "
                "local conditions and management may change actual yield."
            ),
        )

    def _market_assessment(
        self,
        market: Optional[MarketResearchOutput],
        warnings: List[str],
    ) -> MarketAssessment:
        if market is None:
            warnings.append("Market Researcher Agent output is unavailable.")
            return MarketAssessment(
                status="not_available",
                interpretation="No market report was supplied.",
            )
        warnings.append(
            "Market prices, demand, and profitability are estimates; verify current local buyers and prices."
        )
        return MarketAssessment(
            status="available",
            region=market.region,
            recommendations=market.recommendations,
            interpretation=(
                "Market Researcher results are estimates for comparison, "
                "not a guarantee of price, demand, or profit."
            ),
        )

    def _sustainability_actions(
        self,
        farmer_recommendations: List[dict[str, Any]],
        weather: Optional[WeatherResponse],
    ) -> List[AdvisoryAction]:
        actions: List[AdvisoryAction] = []
        for recommendation in farmer_recommendations:
            crop = str(recommendation.get("crop", "the crop"))
            metrics = recommendation.get("sustainability_metrics") or {}
            if hasattr(metrics, "model_dump"):
                metrics = metrics.model_dump()

            water_efficiency = metrics.get("water_efficiency")
            if isinstance(water_efficiency, (int, float)) and water_efficiency < 0.5:
                evidence = [
                    f"Farmer Advisor water_efficiency={water_efficiency:.2f} for {crop}."
                ]
                if weather and any(
                    risk.type in {"heavy_rain", "sustained_rainfall"}
                    for risk in weather.agricultural_risks
                ):
                    evidence.append(
                        "Weather Agent reports a heavy-rain or sustained-rainfall risk."
                    )
                    action = (
                        "Check soil moisture before irrigation and inspect field drainage; "
                        "avoid irrigation while the soil is already wet."
                    )
                else:
                    action = (
                        "Monitor soil moisture and use irrigation only when the crop needs it."
                    )
                actions.append(AdvisoryAction(
                    action=action,
                    rationale="The supplied Farmer Advisor metrics indicate lower water efficiency.",
                    priority="medium",
                    evidence=evidence,
                ))

            soil_health = metrics.get("soil_health")
            if isinstance(soil_health, (int, float)) and soil_health < 0.5:
                actions.append(AdvisoryAction(
                    action="Consider locally suitable crop rotation or cover-cropping practices.",
                    rationale="The supplied Farmer Advisor metrics indicate lower soil-health suitability.",
                    priority="low",
                    evidence=[
                        f"Farmer Advisor soil_health={soil_health:.2f} for {crop}."
                    ],
                ))
        return actions

    def _recommended_actions(
        self,
        disease: Optional[DiseasePredictionResponse],
        weather: Optional[WeatherResponse],
        market: Optional[MarketResearchOutput],
        yield_prediction: Optional[YieldPredictionResponse],
    ) -> List[AdvisoryAction]:
        actions: List[AdvisoryAction] = []
        if disease and disease.disease.strip().lower() not in {"healthy", "normal"}:
            priority = (
                "high" if disease.confidence >= 0.85
                else "medium" if disease.confidence >= 0.6
                else "low"
            )
            actions.append(AdvisoryAction(
                action=(
                    "Inspect the affected plant and nearby plants, and confirm with a local "
                    "agricultural expert before applying treatment."
                ),
                rationale=(
                    "The disease classifier reported a non-healthy class; image classification "
                    "alone does not confirm disease."
                ),
                priority=priority,
                evidence=[
                    f"Disease Agent: {disease.crop} / {disease.disease}",
                    f"Model confidence: {disease.confidence:.3f}",
                ],
            ))

        if weather:
            for risk in weather.agricultural_risks:
                if risk.type in {"heat_stress", "high_temperature"}:
                    actions.append(AdvisoryAction(
                        action=(
                            "Check crop condition and soil moisture; irrigate only if needed "
                            "and water is available."
                        ),
                        rationale="Weather Agent reported elevated-temperature risk.",
                        priority=risk.severity if risk.severity in {"low", "medium", "high"} else "medium",
                        evidence=[f"Weather Agent risk: {risk.type} ({risk.severity})."],
                    ))
                elif risk.type in {"heavy_rain", "sustained_rainfall"}:
                    actions.append(AdvisoryAction(
                        action=(
                            "Inspect drainage and postpone unnecessary irrigation or field work "
                            "if conditions are waterlogged."
                        ),
                        rationale="Weather Agent reported a rainfall-related risk.",
                        priority=risk.severity if risk.severity in {"low", "medium", "high"} else "medium",
                        evidence=[f"Weather Agent risk: {risk.type} ({risk.severity})."],
                    ))
                elif risk.type == "high_humidity":
                    actions.append(AdvisoryAction(
                        action="Scout crops for disease symptoms; do not apply pesticide based on humidity alone.",
                        rationale="Weather Agent reported high humidity, which may increase disease pressure.",
                        priority="low",
                        evidence=[f"Weather Agent risk: {risk.type} ({risk.severity})."],
                    ))

        if market:
            actions.append(AdvisoryAction(
                action="Verify current local prices and buyer demand before making planting or sales commitments.",
                rationale="The market report is an estimate and may not reflect a farmer's actual local transaction.",
                priority="medium",
                evidence=[f"Market Researcher region: {market.region}."],
            ))

        if yield_prediction:
            actions.append(AdvisoryAction(
                action="Use the yield estimate as one planning input and compare it with farm records and local advice.",
                rationale="The model output includes uncertainty and is not a guaranteed harvest.",
                priority="low",
                evidence=[
                    f"Yield model: {yield_prediction.model_version}.",
                    f"Prediction range metadata: {yield_prediction.confidence_or_range}",
                ],
            ))
        return actions
