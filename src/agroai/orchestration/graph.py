from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from ...agents.disease_agent import DiseaseAgent
from ...agents.farmer_advisor import FarmerAdvisor, FarmerProfile
from ...agents.local_advisory_agent import (
    CropCandidateAssessment,
    LocalAdvisoryAgent,
    LocalAdvisoryInput,
)
from ...agents.market_researcher import (
    MarketResearchInput,
    MarketResearcher,
)
from ...agents.weather_agent import WeatherAgent
from ..agents.yield_prediction import (
    YieldPredictionAgent,
    YieldPredictionRequest,
)
from .state import (
    AgentError,
    AgroAIRequest,
    AgroAIResponse,
    AgroAIState,
    SoilInformation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPECIALIZED_AGENTS = (
    "farmer_advisor",
    "weather",
    "disease",
    "yield_prediction",
    "market_researcher",
)


class AgroAIOrchestrator:
    """Conditional LangGraph orchestration around the existing independent agents."""

    def __init__(
        self,
        farmer_advisor: Optional[FarmerAdvisor] = None,
        market_researcher: Optional[MarketResearcher] = None,
        weather_agent: Optional[WeatherAgent] = None,
        disease_agent: Optional[DiseaseAgent] = None,
        yield_prediction_agent: Optional[YieldPredictionAgent] = None,
        local_advisory_agent: Optional[LocalAdvisoryAgent] = None,
    ) -> None:
        self._agents: Dict[str, Any] = {
            "farmer_advisor": farmer_advisor,
            "market_researcher": market_researcher,
            "weather": weather_agent,
            "disease": disease_agent,
            "yield_prediction": yield_prediction_agent,
            "local_advisory": local_advisory_agent or LocalAdvisoryAgent(),
        }
        self.graph = self._build_graph()

    async def ainvoke(
        self,
        request: AgroAIRequest | Dict[str, Any],
    ) -> AgroAIResponse:
        payload = (
            request.model_dump()
            if isinstance(request, AgroAIRequest)
            else request
        )
        result = await self.graph.ainvoke({"input_payload": payload})
        return AgroAIResponse.model_validate(result["response"])

    def _build_graph(self):
        builder = StateGraph(AgroAIState)
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("determine_required_agents", self._determine_required_agents)
        builder.add_node("farmer_advisor", self._run_farmer_advisor)
        builder.add_node("weather", self._run_weather)
        builder.add_node("disease", self._run_disease)
        builder.add_node("yield_prediction", self._run_yield_prediction)
        builder.add_node("market_researcher", self._run_market_researcher)
        builder.add_node("combine_outputs", self._combine_outputs)
        builder.add_node("local_advisory", self._run_local_advisory)
        builder.add_node("validate_final_response", self._validate_final_response)

        builder.add_edge(START, "validate_input")
        builder.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {
                "determine_required_agents": "determine_required_agents",
                "validate_final_response": "validate_final_response",
            },
        )
        builder.add_conditional_edges(
            "determine_required_agents",
            self._route_to_agents,
        )
        for agent_name in SPECIALIZED_AGENTS:
            builder.add_edge(agent_name, "combine_outputs")
        builder.add_edge("combine_outputs", "local_advisory")
        builder.add_edge("local_advisory", "validate_final_response")
        builder.add_edge("validate_final_response", END)
        return builder.compile()

    def _validate_input(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            request = AgroAIRequest.model_validate(state.get("input_payload", {}))
            self._validate_intent_requirements(request)
        except (ValidationError, ValueError) as exc:
            message = (
                "Input validation failed"
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            intent = state.get("input_payload", {}).get("intent", "advisory")
            if intent not in {
                "yield_only",
                "disease_detection",
                "market_analysis",
                "weather",
                "farm_assessment",
                "advisory",
            }:
                intent = "advisory"
            return {
                "input_valid": False,
                "input_error": message,
                "intent": intent,
                "agent_errors": [
                    AgentError(
                        agent="input",
                        error_type=type(exc).__name__,
                        message=message,
                    )
                ],
            }

        data = request.model_dump()
        return {
            **data,
            "input_valid": True,
            "farmer_profile": request.farmer_profile,
            "soil": request.soil,
            "executed_agents": [],
            "agent_errors": [],
        }

    def _validate_intent_requirements(self, request: AgroAIRequest) -> None:
        crop = self._effective_crop(
            request.crop,
            request.farmer_profile,
        )
        location = self._effective_location(
            request.location,
            request.farmer_profile,
        )
        crops = self._effective_market_crops(request)
        farm_size = request.farm_size or (
            request.farmer_profile.farm_size
            if request.farmer_profile
            else None
        )

        if request.intent == "yield_only" and not crop:
            raise ValueError("A crop or farmer preferred crop is required for yield prediction")
        if request.intent == "disease_detection" and not request.image:
            raise ValueError("A crop image is required for disease detection")
        if request.intent == "weather" and not (
            location or (request.latitude is not None and request.longitude is not None)
        ):
            raise ValueError("A location or both coordinates are required for weather")
        if request.intent == "market_analysis":
            if not crops:
                raise ValueError("At least one crop is required for market analysis")
            if not location:
                raise ValueError("A location or region is required for market analysis")
            if not farm_size:
                raise ValueError("A positive farm size is required for market analysis")
        if request.intent == "farm_assessment" and request.farmer_profile is None:
            raise ValueError("A farmer profile is required for a full farm assessment")

    def _route_after_validation(self, state: AgroAIState) -> str:
        return (
            "determine_required_agents"
            if state.get("input_valid")
            else "validate_final_response"
        )

    def _determine_required_agents(self, state: AgroAIState) -> Dict[str, Any]:
        intent = state["intent"]
        profile = state.get("farmer_profile")
        crop = self._effective_crop(state.get("crop"), profile)
        location = self._effective_location(state.get("location"), profile)
        latitude = state.get("latitude")
        longitude = state.get("longitude")
        has_location = bool(location) or (latitude is not None and longitude is not None)
        has_image = bool(state.get("image"))
        market_crops = self._effective_market_crops_from_state(state)
        farm_size = state.get("farm_size") or (profile.farm_size if profile else None)

        selected: List[str] = []
        skipped: Dict[str, str] = {}

        def select(agent: str, available: bool, reason: str) -> None:
            if available:
                selected.append(agent)
            else:
                skipped[agent] = reason

        if intent == "yield_only":
            select("yield_prediction", bool(crop), "A crop was not provided.")
            for name in ("farmer_advisor", "weather", "disease", "market_researcher"):
                skipped[name] = "Not requested for a yield-only analysis."
        elif intent == "disease_detection":
            select("disease", has_image, "A crop image was not provided.")
            select("weather", has_location, "No location or complete coordinates were provided.")
            for name in ("farmer_advisor", "yield_prediction", "market_researcher"):
                skipped[name] = "Not requested for disease detection."
        elif intent == "weather":
            select("weather", has_location, "No location or complete coordinates were provided.")
            for name in ("farmer_advisor", "disease", "yield_prediction", "market_researcher"):
                skipped[name] = "Not requested for weather analysis."
        elif intent == "market_analysis":
            select("market_researcher", bool(market_crops and location and farm_size), "Market inputs are incomplete.")
            for name in ("farmer_advisor", "weather", "disease", "yield_prediction"):
                skipped[name] = "Not requested for market analysis."
        elif intent == "farm_assessment":
            select("farmer_advisor", profile is not None, "A farmer profile was not provided.")
            select("weather", has_location, "No location or complete coordinates were provided.")
            select("disease", has_image, "A disease image was not provided.")
            select("yield_prediction", bool(crop), "No crop was selected for yield prediction.")
            select(
                "market_researcher",
                bool(market_crops and location and farm_size),
                "Crop, location, or farm-size input is missing.",
            )
        else:
            select("farmer_advisor", profile is not None, "A farmer profile was not provided.")
            select("weather", has_location, "No location or complete coordinates were provided.")
            select("disease", has_image, "A disease image was not provided.")
            select("yield_prediction", bool(crop), "No crop was selected for yield prediction.")
            select(
                "market_researcher",
                bool(market_crops and location and farm_size),
                "Crop, location, or farm-size input is missing.",
            )

        return {
            "required_agents": selected,
            "skipped_agents": skipped,
            "crop": crop,
            "location": location,
            "farm_size": farm_size,
            "market_crops": market_crops,
        }

    def _route_to_agents(self, state: AgroAIState) -> Sequence[str]:
        selected = state.get("required_agents", [])
        return selected if selected else ["combine_outputs"]

    async def _run_farmer_advisor(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            advisor = self._get_agent("farmer_advisor")
            if advisor is None:
                frame = self._load_dataset("farmer_advisor_dataset.csv")
                advisor = FarmerAdvisor(frame)
            recommendations = await asyncio.to_thread(
                advisor.get_crop_recommendations,
                state["farmer_profile"],
            )
            recommendations = [
                {
                    key: value.model_dump() if isinstance(value, BaseModel) else value
                    for key, value in recommendation.items()
                }
                for recommendation in recommendations
            ]
            return {
                "farmer_recommendations": recommendations,
                "executed_agents": ["farmer_advisor"],
            }
        except Exception as exc:
            return self._agent_failure("farmer_advisor", exc)

    async def _run_market_researcher(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            researcher = self._get_agent("market_researcher")
            if researcher is None:
                researcher = MarketResearcher(
                    self._load_dataset("market_researcher_dataset.csv")
                )
            request = MarketResearchInput(
                crops=state["market_crops"],
                region=state["location"],
                farm_size=state["farm_size"],
            )
            result = await asyncio.to_thread(researcher.run, request)
            return {
                "market_analysis": result,
                "executed_agents": ["market_researcher"],
            }
        except Exception as exc:
            return self._agent_failure("market_researcher", exc)

    async def _run_weather(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            agent = self._get_agent("weather") or WeatherAgent()
            result = await agent.get_weather(
                location=state.get("location"),
                latitude=state.get("latitude"),
                longitude=state.get("longitude"),
            )
            return {
                "weather": result,
                "executed_agents": ["weather"],
            }
        except Exception as exc:
            return self._agent_failure("weather", exc)

    async def _run_disease(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            agent = self._get_agent("disease") or DiseaseAgent()
            result = await asyncio.to_thread(
                agent.predict,
                state["image"],
                crop_hint=state.get("image_crop") or state.get("crop"),
            )
            return {
                "disease": result,
                "executed_agents": ["disease"],
            }
        except Exception as exc:
            return self._agent_failure("disease", exc)

    async def _run_yield_prediction(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            agent = self._get_agent("yield_prediction") or YieldPredictionAgent()
            soil = state.get("soil") or SoilInformation()
            request = YieldPredictionRequest(
                crop=state["crop"],
                location=state.get("location"),
                soil_ph=soil.soil_ph,
                soil_moisture=soil.soil_moisture,
                temperature_c=state.get("temperature_c"),
                rainfall_mm=state.get("rainfall_mm"),
                fertilizer_usage_kg=state.get("fertilizer_usage_kg"),
                pesticide_usage_kg=state.get("pesticide_usage_kg"),
                sustainability_score=state.get("sustainability_score"),
                historical_yield=state.get("historical_yield"),
            )
            result = await asyncio.to_thread(agent.predict, request)
            return {
                "yield_prediction": result,
                "executed_agents": ["yield_prediction"],
            }
        except Exception as exc:
            return self._agent_failure("yield_prediction", exc)

    def _combine_outputs(self, state: AgroAIState) -> Dict[str, Any]:
        recommendations = state.get("farmer_recommendations", [])
        return {
            "sustainability_assessment": [
                CropCandidateAssessment.model_validate(item)
                for item in recommendations
            ],
        }

    async def _run_local_advisory(self, state: AgroAIState) -> Dict[str, Any]:
        try:
            agent = self._get_agent("local_advisory")
            request = LocalAdvisoryInput(
                farmer_profile=state.get("farmer_profile"),
                farmer_recommendations=state.get("farmer_recommendations", []),
                market_research=state.get("market_analysis"),
                weather=state.get("weather"),
                disease=state.get("disease"),
                yield_prediction=state.get("yield_prediction"),
            )
            result = await asyncio.to_thread(agent.run, request)
            return {
                "final_advisory": result,
                "executed_agents": ["local_advisory"],
            }
        except Exception as exc:
            return self._agent_failure("local_advisory", exc)

    def _validate_final_response(self, state: AgroAIState) -> Dict[str, Any]:
        errors = list(state.get("agent_errors", []))
        if state.get("input_error"):
            skipped = {
                name: "Input validation failed; agent was not called."
                for name in (*SPECIALIZED_AGENTS, "local_advisory")
            }
        else:
            skipped = state.get("skipped_agents", {})

        response_data = {
            "intent": state.get("intent", "advisory"),
            "executed_agents": state.get("executed_agents", []),
            "skipped_agents": skipped,
            "agent_errors": errors,
            "farmer_profile": state.get("farmer_profile"),
            "crop": state.get("crop"),
            "location": state.get("location"),
            "farm_size": state.get("farm_size"),
            "soil": state.get("soil", SoilInformation()),
            "farmer_recommendations": state.get("farmer_recommendations", []),
            "weather": state.get("weather"),
            "disease": state.get("disease"),
            "yield_prediction": state.get("yield_prediction"),
            "market_analysis": state.get("market_analysis"),
            "sustainability_assessment": state.get("sustainability_assessment", []),
            "final_advisory": state.get("final_advisory"),
        }
        try:
            response = AgroAIResponse.model_validate(response_data)
        except ValidationError as exc:
            error = AgentError(
                agent="orchestration",
                error_type="FinalResponseValidationError",
                message="The combined response failed schema validation.",
            )
            response_data["agent_errors"] = [*errors, error]
            response_data["farmer_recommendations"] = []
            response_data["sustainability_assessment"] = []
            response_data["weather"] = None
            response_data["disease"] = None
            response_data["yield_prediction"] = None
            response_data["market_analysis"] = None
            response_data["final_advisory"] = None
            response = AgroAIResponse.model_validate(response_data)
            return {
                "agent_errors": [error],
                "response": response,
            }
        return {"response": response}

    def _get_agent(self, name: str) -> Any:
        return self._agents[name]

    @staticmethod
    def _agent_failure(agent: str, exc: Exception) -> Dict[str, Any]:
        return {
            "executed_agents": [agent],
            "agent_errors": [
                AgentError(
                    agent=agent,
                    error_type=type(exc).__name__,
                    message=f"{agent} agent execution failed; remaining agents will continue.",
                )
            ]
        }

    @staticmethod
    def _load_dataset(filename: str) -> pd.DataFrame:
        candidates = [
            PROJECT_ROOT / "data" / filename,
            PROJECT_ROOT / filename,
        ]
        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(f"Dataset not found: {filename}")

    @staticmethod
    def _effective_crop(
        crop: Optional[str],
        profile: Optional[FarmerProfile],
    ) -> Optional[str]:
        if crop and crop.strip():
            return crop.strip()
        if profile and profile.preferred_crops:
            return profile.preferred_crops[0].strip() or None
        return None

    @staticmethod
    def _effective_location(
        location: Optional[str],
        profile: Optional[FarmerProfile],
    ) -> Optional[str]:
        if location and location.strip():
            return location.strip()
        if profile and profile.location.strip():
            return profile.location.strip()
        return None

    @classmethod
    def _effective_market_crops(cls, request: AgroAIRequest) -> List[str]:
        requested = request.market_crops or []
        if requested:
            return [crop.strip() for crop in requested if crop.strip()]
        crop = cls._effective_crop(request.crop, request.farmer_profile)
        if crop:
            return [crop]
        return []

    @classmethod
    def _effective_market_crops_from_state(
        cls,
        state: AgroAIState,
    ) -> List[str]:
        requested = state.get("market_crops") or []
        if requested:
            return [crop.strip() for crop in requested if crop.strip()]
        crop = cls._effective_crop(state.get("crop"), state.get("farmer_profile"))
        return [crop] if crop else []


def build_agroai_graph(**agent_overrides: Any):
    """Build a compiled graph; overrides make agent execution easy to test."""
    return AgroAIOrchestrator(**agent_overrides).graph
