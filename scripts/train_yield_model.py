from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agroai.data.yield_pipeline import save_yield_artifact, train_yield_models


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "farmer_advisor_dataset.csv"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "farmer_advisor_dataset.csv"

    artifact_path = PROJECT_ROOT / "artifacts" / "yield_model.joblib"
    frame = pd.read_csv(dataset_path)
    artifact = train_yield_models(frame)
    save_yield_artifact(artifact, artifact_path)

    print(f"Dataset: {dataset_path}")
    print(f"Artifact: {artifact_path}")
    print(f"Baseline validation metrics: {artifact['baseline_metrics']}")
    print(f"XGBoost validation metrics: {artifact['validation_metrics']}")
    print(f"XGBoost test metrics: {artifact['test_metrics']}")


if __name__ == "__main__":
    main()
