from __future__ import annotations

import argparse

from config import ARTIFACT_DIR, DATA_PATH, METRICS_PATH, REGISTRY_PATH
from forecast_service import ForecastingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare forecasting models per state.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the Excel dataset")
    args = parser.parse_args()

    service = ForecastingService(data_path=args.data)
    service.build_state_registry()
    summary = service.training_summary()
    print("Training complete")
    print(summary.to_string(index=False))
    print()
    print(f"Artifacts saved in: {ARTIFACT_DIR}")
    print(f"Registry file: {REGISTRY_PATH}")
    print(f"Metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()
