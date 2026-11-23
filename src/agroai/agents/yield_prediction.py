from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import shap

from ..data.yield_pipeline import engineer_features, load_yield_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "yield_model.joblib"


class YieldPredictionRequest(BaseModel):
    crop: str = Field(min_length=1)
    location: Optional[str] = None
    soil_ph: Optional[float] = Field(default=None, ge=0, le=14)
    soil_moisture: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = None
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    fertilizer_usage_kg: Optional[float] = Field(default=None, ge=0)
    pesticide_usage_kg: Optional[float] = Field(default=None, ge=0)
    sustainability_score: Optional[float] = Field(default=None, ge=0, le=100)
    historical_yield: Optional[float] = Field(default=None, ge=0)


class ImportantFactor(BaseModel):
    factor: str
    impact: float


class YieldPredictionResponse(BaseModel):
    crop: str
    location: Optional[str] = None
    predicted_yield: float
    unit: str = "tons"
    confidence_or_range: Dict[str, object]
    important_factors: List[ImportantFactor]
    model_version: str
    historical_yield: Optional[float] = None


class YieldPredictionAgent:
    def __init__(self, artifact_path: Optional[Path] = None) -> None:
        self.artifact_path = artifact_path or DEFAULT_ARTIFACT
        self.artifact = load_yield_artifact(self.artifact_path)
        self.model = self.artifact["model"]
        self.crop_values = set(self.artifact["crop_values"])

    def _build_input(self, request: YieldPredictionRequest) -> pd.DataFrame:
        crop = request.crop.strip().title()
        medians = self.artifact["model_medians"]
        raw = {
            "Crop_Type": crop,
            "Soil_pH": request.soil_ph if request.soil_ph is not None else medians["Soil_pH"],
            "Soil_Moisture": (
                request.soil_moisture
                if request.soil_moisture is not None
                else medians["Soil_Moisture"]
            ),
            "Temperature_C": (
                request.temperature_c
                if request.temperature_c is not None
                else medians["Temperature_C"]
            ),
            "Rainfall_mm": (
                request.rainfall_mm
                if request.rainfall_mm is not None
                else medians["Rainfall_mm"]
            ),
            "Fertilizer_Usage_kg": (
                request.fertilizer_usage_kg
                if request.fertilizer_usage_kg is not None
                else medians["Fertilizer_Usage_kg"]
            ),
            "Pesticide_Usage_kg": (
                request.pesticide_usage_kg
                if request.pesticide_usage_kg is not None
                else medians["Pesticide_Usage_kg"]
            ),
            "Sustainability_Score": (
                request.sustainability_score
                if request.sustainability_score is not None
                else medians["Sustainability_Score"]
            ),
        }
        frame = pd.DataFrame([raw])
        return engineer_features(frame)

    def _important_factors(self, engineered_input: pd.DataFrame) -> List[ImportantFactor]:
        preprocessor = self.model.named_steps["preprocessor"]
        xgb_model = self.model.named_steps["model"]
        transformed = preprocessor.transform(engineered_input)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(transformed)
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        numeric_names = [
            "Soil_pH",
            "Soil_Moisture",
            "Temperature_C",
            "Rainfall_mm",
            "Fertilizer_Usage_kg",
            "Pesticide_Usage_kg",
            "Sustainability_Score",
            "Rainfall_Temperature",
            "Moisture_SoilpH",
            "Fertilizer_per_Rainfall",
        ]
        try:
            crop_encoder = preprocessor.named_transformers_["crop"].named_steps["encoder"]
            crop_names = crop_encoder.get_feature_names_out(["Crop_Type"]).tolist()
        except (AttributeError, KeyError):
            crop_names = []

        feature_names = numeric_names + crop_names
        values = np.asarray(shap_values)
        if values.ndim == 1:
            impacts = values
        else:
            impacts = values[0]

        ranked = sorted(
            zip(feature_names, impacts),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:5]
        return [
            ImportantFactor(
                factor=str(name),
                impact=round(float(value), 6),
            )
            for name, value in ranked
        ]

    def predict(self, request: YieldPredictionRequest) -> YieldPredictionResponse:
        crop = request.crop.strip().title()
        if crop not in self.crop_values:
            raise ValueError(
                f"Unsupported crop '{crop}'. "
                f"Available crops: {sorted(self.crop_values)}"
            )

        engineered_input = self._build_input(request)
        predicted = float(self.model.predict(engineered_input)[0])
        predicted = max(predicted, 0.0)
        residual_std = float(self.artifact["residual_std"])
        lower = max(predicted - 1.96 * residual_std, 0.0)
        upper = predicted + 1.96 * residual_std

        test_metrics = self.artifact["test_metrics"]
        return YieldPredictionResponse(
            crop=crop,
            location=request.location,
            predicted_yield=round(predicted, 4),
            unit="tons",
            confidence_or_range={
                "type": "empirical_95_percent_residual_range",
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "test_rmse": test_metrics["rmse"],
            },
            important_factors=self._important_factors(engineered_input),
            model_version="xgboost-v1",
            historical_yield=request.historical_yield,
        )
