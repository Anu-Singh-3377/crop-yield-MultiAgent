from __future__ import annotations

from operator import add
from typing import Annotated, Any, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from ...agents.disease_agent import DiseasePredictionResponse
from ...agents.farmer_advisor import FarmerProfile
from ...agents.local_advisory_agent import (
    CropCandidateAssessment,
    LocalAdvisoryOutput,
)
from ...agents.market_researcher import MarketResearchOutput
from ...agents.weather_agent import WeatherResponse
from ..agents.yield_prediction import YieldPredictionResponse


AgroAIIntent = Literal[
    "yield_only",
    "disease_detection",
    "market_analysis",
    "weather",
    "farm_assessment",
    "advisory",
]

AgentName = Literal[
    "farmer_advisor",
    "market_researcher",
    "weather",
    "disease",
    "yield_prediction",
    "local_advisory",
    "input",
    "orchestration",
]


class SoilInformation(BaseModel):
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = Field(default=None, ge=0, le=14)
    soil_moisture: Optional[float] = Field(default=None, ge=0)
    water_availability: Optional[str] = None


class AgroAIRequest(BaseModel):
    intent: AgroAIIntent
    farmer_profile: Optional[FarmerProfile] = None
    crop: Optional[str] = Field(default=None, min_length=1)
    market_crops: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    farm_size: Optional[float] = Field(default=None, gt=0)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    soil: SoilInformation = Field(default_factory=SoilInformation)
    temperature_c: Optional[float] = None
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    fertilizer_usage_kg: Optional[float] = Field(default=None, ge=0)
    pesticide_usage_kg: Optional[float] = Field(default=None, ge=0)
    sustainability_score: Optional[float] = Field(default=None, ge=0, le=100)
    historical_yield: Optional[float] = Field(default=None, ge=0)
    image: Optional[bytes] = None
    image_crop: Optional[str] = None


class AgentError(BaseModel):
    agent: AgentName
    error_type: str
    message: str


class AgroAIResponse(BaseModel):
    intent: AgroAIIntent
    executed_agents: List[str] = Field(default_factory=list)
    skipped_agents: dict[str, str] = Field(default_factory=dict)
    agent_errors: List[AgentError] = Field(default_factory=list)
    farmer_profile: Optional[FarmerProfile] = None
    crop: Optional[str] = None
    location: Optional[str] = None
    farm_size: Optional[float] = None
    soil: SoilInformation = Field(default_factory=SoilInformation)
    farmer_recommendations: List[dict[str, Any]] = Field(default_factory=list)
    weather: Optional[WeatherResponse] = None
    disease: Optional[DiseasePredictionResponse] = None
    yield_prediction: Optional[YieldPredictionResponse] = None
    market_analysis: Optional[MarketResearchOutput] = None
    sustainability_assessment: List[CropCandidateAssessment] = Field(default_factory=list)
    final_advisory: Optional[LocalAdvisoryOutput] = None


class AgroAIState(TypedDict, total=False):
    input_payload: dict[str, Any]
    input_valid: bool
    input_error: Optional[str]
    intent: AgroAIIntent
    farmer_profile: Optional[FarmerProfile]
    crop: Optional[str]
    market_crops: List[str]
    location: Optional[str]
    farm_size: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    soil: SoilInformation
    temperature_c: Optional[float]
    rainfall_mm: Optional[float]
    fertilizer_usage_kg: Optional[float]
    pesticide_usage_kg: Optional[float]
    sustainability_score: Optional[float]
    historical_yield: Optional[float]
    image: Optional[bytes]
    image_crop: Optional[str]
    required_agents: List[str]
    skipped_agents: dict[str, str]
    executed_agents: Annotated[List[str], add]
    farmer_recommendations: List[dict[str, Any]]
    weather: Optional[WeatherResponse]
    disease: Optional[DiseasePredictionResponse]
    yield_prediction: Optional[YieldPredictionResponse]
    market_analysis: Optional[MarketResearchOutput]
    sustainability_assessment: List[CropCandidateAssessment]
    final_advisory: Optional[LocalAdvisoryOutput]
    agent_errors: Annotated[List[AgentError], add]
    response: AgroAIResponse
