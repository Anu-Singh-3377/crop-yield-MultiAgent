from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


CROP_COLUMN = "Crop_Type"
TARGET_COLUMN = "Crop_Yield_ton"
NUMERIC_FEATURES = [
    "Soil_pH",
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Fertilizer_Usage_kg",
    "Pesticide_Usage_kg",
    "Sustainability_Score",
]
CATEGORICAL_FEATURES = [CROP_COLUMN]

REQUIRED_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN]


def validate_yield_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required yield dataset columns: {missing}")

    validated = frame[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    validated[CROP_COLUMN] = validated[CROP_COLUMN].astype("string").str.strip().str.title()
    validated = validated.dropna(subset=[CROP_COLUMN, TARGET_COLUMN])
    validated = validated[np.isfinite(validated[TARGET_COLUMN])]

    if validated.empty:
        raise ValueError("Yield dataset has no usable rows")
    return validated


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    engineered["Rainfall_Temperature"] = (
        engineered["Rainfall_mm"] * engineered["Temperature_C"]
    )
    engineered["Moisture_SoilpH"] = (
        engineered["Soil_Moisture"] * engineered["Soil_pH"]
    )
    engineered["Fertilizer_per_Rainfall"] = (
        engineered["Fertilizer_Usage_kg"] /
        engineered["Rainfall_mm"].replace(0, np.nan)
    )
    return engineered


def split_dataset(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    features = engineer_features(frame)
    target = features.pop(TARGET_COLUMN)

    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=42,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = NUMERIC_FEATURES + [
        "Rainfall_Temperature",
        "Moisture_SoilpH",
        "Fertilizer_per_Rainfall",
    ]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("crop", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_yield_models(
    frame: pd.DataFrame,
) -> Dict[str, object]:
    validated = validate_yield_data(frame)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(validated)
    preprocessor = build_preprocessor()

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])
    baseline.fit(x_train, y_train)
    baseline_validation = evaluate_regression(
        y_validation,
        baseline.predict(x_validation),
    )

    xgb_preprocessor = build_preprocessor()
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    xgb = Pipeline([
        ("preprocessor", xgb_preprocessor),
        ("model", xgb_model),
    ])
    xgb.fit(x_train, y_train)

    validation_prediction = xgb.predict(x_validation)
    test_prediction = xgb.predict(x_test)
    validation_metrics = evaluate_regression(y_validation, validation_prediction)
    test_metrics = evaluate_regression(y_test, test_prediction)

    test_residuals = np.asarray(y_test) - test_prediction
    residual_std = float(np.std(test_residuals))
    model_medians = {
        column: float(validated[column].median())
        for column in NUMERIC_FEATURES
    }

    artifact = {
        "model": xgb,
        "baseline_metrics": baseline_validation,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "residual_std": residual_std,
        "model_medians": model_medians,
        "crop_values": sorted(validated[CROP_COLUMN].dropna().unique().tolist()),
        "feature_schema": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "engineered": [
                "Rainfall_Temperature",
                "Moisture_SoilpH",
                "Fertilizer_per_Rainfall",
            ],
        },
    }
    return artifact


def save_yield_artifact(artifact: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def load_yield_artifact(input_path: Path) -> Dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Yield model artifact not found: {input_path}. "
            "Run scripts/train_yield_model.py first."
        )
    return joblib.load(input_path)
