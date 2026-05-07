from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REGISTRY_PATH = ARTIFACT_DIR / "forecast_registry.pkl"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.csv"


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 8
    validation_size: int = 8
    freq: str = "W-SAT"
    seasonal_period: int = 52
    lookback: int = 12
    random_state: int = 42
    rolling_windows: tuple[int, int] = (4, 8)
    lag_features: tuple[int, int, int] = (1, 7, 30)
    sarima_orders: tuple[tuple[int, int, int], ...] = (
        (0, 1, 1),
        (1, 1, 1),
    )
    sarima_seasonal_orders: tuple[tuple[int, int, int], ...] = (
        (0, 1, 1),
        (1, 1, 0),
    )
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1
    lstm_epochs: int = 15
    lstm_patience: int = 2
    lstm_learning_rate: float = 0.01
