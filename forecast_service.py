from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from xgboost import XGBRegressor

from config import ARTIFACT_DIR, DATA_PATH, METRICS_PATH, REGISTRY_PATH, ForecastConfig
from data_prep import (
    add_lag_and_rolling_features,
    dataframe_to_json_ready,
    future_dates,
    get_state_names,
    json_ready_to_dataframe,
    load_sales_data,
    make_supervised_frame,
    make_weekly_frame,
    train_validation_split,
)

warnings.filterwarnings("ignore")


FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "quarter",
    "weekofyear",
    "holiday_flag",
    "trend_idx",
    "is_month_start",
    "is_month_end",
    "lag_1",
    "lag_7",
    "lag_30",
    "roll_mean_4",
    "roll_std_4",
    "roll_mean_8",
    "roll_std_8",
    "ewm_4",
    "ewm_8",
]


class ForecastLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


class ModelScore:
    def __init__(self, model_name: str, mae: float, rmse: float, mape: float):
        self.model_name = model_name
        self.mae = mae
        self.rmse = rmse
        self.mape = mape


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelScore:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _safe_mape(y_true, y_pred)
    return ModelScore("", mae, rmse, mape)


def _score_dict(score: ModelScore) -> dict[str, float]:
    return {"mae": score.mae, "rmse": score.rmse, "mape": score.mape}


def _holiday_flag_for_date(date: pd.Timestamp) -> int:
    import holidays

    us_holidays = holidays.US(years=[date.year])
    return int(date.date() in us_holidays)


def _build_recursive_feature_row(history_frame: pd.DataFrame, next_date: pd.Timestamp, cfg: ForecastConfig) -> dict[str, Any]:
    values = history_frame["Total"].astype(float).to_numpy()
    row: dict[str, Any] = {
        "Date": next_date,
        "State": history_frame["State"].iloc[-1],
        "Category": history_frame["Category"].iloc[-1],
        "holiday_flag": _holiday_flag_for_date(next_date),
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "quarter": next_date.quarter,
        "weekofyear": int(next_date.isocalendar().week),
        "trend_idx": int(history_frame["trend_idx"].iloc[-1]) + 1,
        "is_month_start": int(next_date.is_month_start),
        "is_month_end": int(next_date.is_month_end),
    }

    for lag in cfg.lag_features:
        row[f"lag_{lag}"] = float(values[-lag]) if len(values) >= lag else np.nan

    for window in cfg.rolling_windows:
        if len(values) >= window:
            recent = values[-window:]
            row[f"roll_mean_{window}"] = float(np.mean(recent))
            row[f"roll_std_{window}"] = float(np.std(recent, ddof=0))
        else:
            row[f"roll_mean_{window}"] = np.nan
            row[f"roll_std_{window}"] = np.nan

    row["ewm_4"] = float(pd.Series(values).ewm(span=4, adjust=False).mean().iloc[-1]) if len(values) else np.nan
    row["ewm_8"] = float(pd.Series(values).ewm(span=8, adjust=False).mean().iloc[-1]) if len(values) else np.nan
    return row


def _prepare_prophet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prophet_df = frame[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"}).copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    return prophet_df


def _make_prophet_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    import holidays

    holiday_records = []
    for year in range(start_year, end_year + 1):
        us = holidays.US(years=[year])
        for date, name in us.items():
            holiday_records.append({"ds": pd.Timestamp(date), "holiday": str(name)})
    return pd.DataFrame(holiday_records).drop_duplicates()


def _evaluate_sarima(train: pd.DataFrame, valid: pd.DataFrame, cfg: ForecastConfig) -> tuple[ModelScore, Any]:
    best_model = None
    best_score: ModelScore | None = None
    seasonal_period = cfg.seasonal_period if len(train) >= cfg.seasonal_period * 2 else max(4, min(26, len(train) // 2))

    for order in cfg.sarima_orders:
        for seasonal_base in cfg.sarima_seasonal_orders:
            seasonal_order = (*seasonal_base, seasonal_period)
            try:
                model = SARIMAX(
                    train["Total"],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
                pred = fitted.forecast(steps=len(valid)).to_numpy()
                score = _calc_metrics(valid["Total"].to_numpy(), pred)
                score.model_name = "sarima"
                if best_score is None or score.rmse < best_score.rmse:
                    best_score = score
                    best_model = fitted
            except Exception:
                continue

    if best_score is None or best_model is None:
        raise RuntimeError("SARIMA could not be fitted for this state")
    return best_score, best_model


def _fit_sarima_full(frame: pd.DataFrame, cfg: ForecastConfig) -> Any:
    best_model = None
    best_aic = None
    seasonal_period = cfg.seasonal_period if len(frame) >= cfg.seasonal_period * 2 else max(4, min(26, len(frame) // 2))

    for order in cfg.sarima_orders:
        for seasonal_base in cfg.sarima_seasonal_orders:
            seasonal_order = (*seasonal_base, seasonal_period)
            try:
                model = SARIMAX(
                    frame["Total"],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
                if best_aic is None or fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Could not fit SARIMA on full data")
    return best_model


def _evaluate_prophet(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[ModelScore, Prophet]:
    train_df = _prepare_prophet_frame(train)
    future_holidays = _make_prophet_holidays(train["Date"].min().year, valid["Date"].max().year)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        holidays=future_holidays,
    )
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(valid), freq="W-SAT", include_history=False)
    forecast = model.predict(future)
    pred = forecast["yhat"].to_numpy()
    score = _calc_metrics(valid["Total"].to_numpy(), pred)
    score.model_name = "prophet"
    return score, model


def _fit_prophet_full(frame: pd.DataFrame) -> Prophet:
    prophet_df = _prepare_prophet_frame(frame)
    holidays_df = _make_prophet_holidays(frame["Date"].min().year, frame["Date"].max().year)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        holidays=holidays_df,
    )
    model.fit(prophet_df)
    return model


def _train_xgb(train_supervised: pd.DataFrame) -> XGBRegressor:
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        reg_lambda=1.0,
    )
    xgb.fit(train_supervised[FEATURE_COLUMNS], train_supervised["Total"])
    return xgb


def _recursive_xgb_forecast(model: XGBRegressor, history_frame: pd.DataFrame, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    history = history_frame.copy().sort_values("Date").reset_index(drop=True)
    preds: list[float] = []
    dates = future_dates(history["Date"].iloc[-1], horizon, cfg.freq)

    for next_date in dates:
        row = _build_recursive_feature_row(history, next_date, cfg)
        row_df = pd.DataFrame([row])
        pred = float(model.predict(row_df[FEATURE_COLUMNS])[0])
        preds.append(pred)
        row["Total"] = pred
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return np.asarray(preds, dtype=float)


def _evaluate_xgb(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, cfg: ForecastConfig) -> tuple[ModelScore, XGBRegressor]:
    train_supervised = make_supervised_frame(train_frame, cfg)
    if train_supervised.empty:
        raise RuntimeError("XGBoost training frame is empty after feature generation")
    model = _train_xgb(train_supervised)
    pred = _recursive_xgb_forecast(model, train_frame, len(valid_frame), cfg)
    score = _calc_metrics(valid_frame["Total"].to_numpy(), pred)
    score.model_name = "xgboost"
    return score, model


def _fit_xgb_full(frame: pd.DataFrame, cfg: ForecastConfig) -> XGBRegressor:
    supervised = make_supervised_frame(frame, cfg)
    if supervised.empty:
        raise RuntimeError("XGBoost training frame is empty")
    return _train_xgb(supervised)


def _train_lstm(train_values: np.ndarray, cfg: ForecastConfig) -> tuple[ForecastLSTM, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_values.reshape(-1, 1)).astype(np.float32)
    lookback = cfg.lookback

    xs, ys = [], []
    for idx in range(lookback, len(scaled)):
        xs.append(scaled[idx - lookback : idx])
        ys.append(scaled[idx])

    if not xs:
        raise RuntimeError("Not enough observations for LSTM training")

    x_tensor = torch.tensor(np.asarray(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(ys), dtype=torch.float32)

    model = ForecastLSTM(input_size=1, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_learning_rate)
    loss_fn = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    patience = 0

    model.train()
    for _epoch in range(cfg.lstm_epochs):
        optimizer.zero_grad()
        out = model(x_tensor)
        loss = loss_fn(out, y_tensor)
        loss.backward()
        optimizer.step()

        epoch_loss = float(loss.item())
        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            best_state = pickle.dumps(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.lstm_patience:
                break

    if best_state is not None:
        model.load_state_dict(pickle.loads(best_state))
    return model, scaler


def _recursive_lstm_forecast(model: ForecastLSTM, scaler: MinMaxScaler, history_values: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    lookback = cfg.lookback
    scaled_history = scaler.transform(history_values.reshape(-1, 1)).astype(np.float32).flatten().tolist()
    preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            window = scaled_history[-lookback:]
            if len(window) < lookback:
                pad = [window[0] if window else 0.0] * (lookback - len(window))
                window = pad + window
            x = torch.tensor(np.array(window, dtype=np.float32).reshape(1, lookback, 1))
            scaled_pred = float(model(x).item())
            pred = float(scaler.inverse_transform(np.array([[scaled_pred]], dtype=np.float32))[0, 0])
            preds.append(pred)
            scaled_history.append(scaled_pred)
    return np.asarray(preds, dtype=float)


def _evaluate_lstm(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, cfg: ForecastConfig) -> tuple[ModelScore, dict[str, Any]]:
    model, scaler = _train_lstm(train_frame["Total"].to_numpy(dtype=float), cfg)
    pred = _recursive_lstm_forecast(model, scaler, train_frame["Total"].to_numpy(dtype=float), len(valid_frame), cfg)
    score = _calc_metrics(valid_frame["Total"].to_numpy(), pred)
    score.model_name = "lstm"
    payload = {
        "model": model,
        "scaler": scaler,
        "lookback": cfg.lookback,
    }
    return score, payload


def _fit_lstm_full(frame: pd.DataFrame, cfg: ForecastConfig) -> dict[str, Any]:
    model, scaler = _train_lstm(frame["Total"].to_numpy(dtype=float), cfg)
    return {"model": model, "scaler": scaler, "lookback": cfg.lookback}


def _sarima_forecast(model: Any, horizon: int) -> np.ndarray:
    forecast = model.forecast(steps=horizon)
    return np.asarray(forecast, dtype=float)


def _prophet_forecast(model: Prophet, last_date: pd.Timestamp, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    future = model.make_future_dataframe(periods=horizon, freq=cfg.freq, include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy(dtype=float)


def _lstm_forecast(payload: dict[str, Any], frame: pd.DataFrame, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    return _recursive_lstm_forecast(payload["model"], payload["scaler"], frame["Total"].to_numpy(dtype=float), horizon, cfg)


def _xgb_forecast(model: XGBRegressor, frame: pd.DataFrame, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    return _recursive_xgb_forecast(model, frame, horizon, cfg)


class ForecastingService:
    def __init__(self, cfg: ForecastConfig | None = None, data_path: str | Path = DATA_PATH):
        self.cfg = cfg or ForecastConfig()
        self.data_path = Path(data_path)
        self.registry: dict[str, dict[str, Any]] = {}

    def build_state_registry(self) -> dict[str, dict[str, Any]]:
        raw = load_sales_data(self.data_path)
        registry: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        state_names = get_state_names(raw)

        for idx, state in enumerate(state_names):
            print(f"[{idx+1}/{len(state_names)}] Training {state}...", end=" ", flush=True)
            try:
                state_df = raw[raw["State"] == state].copy().reset_index(drop=True)
                weekly_frame = make_weekly_frame(state_df, self.cfg)
                train_base, valid_base = train_validation_split(weekly_frame, self.cfg.validation_size)

                # Train models
                sarima_score, sarima_model = _evaluate_sarima(train_base, valid_base, self.cfg)
                prophet_score, prophet_model = _evaluate_prophet(train_base, valid_base)
                xgb_score, xgb_model = _evaluate_xgb(train_base, valid_base, self.cfg)
                lstm_score, lstm_payload = _evaluate_lstm(train_base, valid_base, self.cfg)

                scores = [sarima_score, prophet_score, xgb_score, lstm_score]
                best = min(scores, key=lambda s: s.rmse)

                # Fit on full data
                full_fit: dict[str, Any] = {
                    "sarima": _fit_sarima_full(weekly_frame, self.cfg),
                    "prophet": _fit_prophet_full(weekly_frame),
                    "xgboost": _fit_xgb_full(weekly_frame, self.cfg),
                    "lstm": _fit_lstm_full(weekly_frame, self.cfg),
                }

                if best.model_name == "sarima":
                    forecast_values = _sarima_forecast(full_fit["sarima"], self.cfg.horizon)
                elif best.model_name == "prophet":
                    forecast_values = _prophet_forecast(full_fit["prophet"], weekly_frame["Date"].iloc[-1], self.cfg.horizon, self.cfg)
                elif best.model_name == "xgboost":
                    forecast_values = _xgb_forecast(full_fit["xgboost"], weekly_frame, self.cfg.horizon, self.cfg)
                else:
                    forecast_values = _lstm_forecast(full_fit["lstm"], weekly_frame, self.cfg.horizon, self.cfg)

                next_dates = future_dates(weekly_frame["Date"].iloc[-1], self.cfg.horizon, self.cfg.freq)
                forecast_df = pd.DataFrame({"Date": next_dates, "Forecast": forecast_values})

                registry[state] = {
                    "state": state,
                    "category": weekly_frame["Category"].iloc[0],
                    "history": weekly_frame,
                    "validation": valid_base,
                    "best_model": best.model_name,
                    "scores": {score.model_name: _score_dict(score) for score in scores},
                    "models": full_fit,
                    "forecast": forecast_df,
                }

                rows.append(
                    {
                        "State": state,
                        "BestModel": best.model_name,
                        "SARIMA_RMSE": sarima_score.rmse,
                        "Prophet_RMSE": prophet_score.rmse,
                        "XGBoost_RMSE": xgb_score.rmse,
                        "LSTM_RMSE": lstm_score.rmse,
                    }
                )
                print("✓")
            except Exception as e:
                print(f"✗ ({str(e)[:50]})")
                continue

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(registry, REGISTRY_PATH)
        pd.DataFrame(rows).sort_values(["BestModel", "State"]).to_csv(METRICS_PATH, index=False)
        self.registry = registry
        return registry

    def load_registry(self) -> dict[str, dict[str, Any]]:
        if REGISTRY_PATH.exists():
            self.registry = joblib.load(REGISTRY_PATH)
            return self.registry
        return self.build_state_registry()

    def ensure_ready(self) -> dict[str, dict[str, Any]]:
        if self.registry:
            return self.registry
        return self.load_registry()

    def list_states(self) -> list[str]:
        return sorted(self.ensure_ready().keys())

    def get_state_result(self, state: str) -> dict[str, Any]:
        registry = self.ensure_ready()
        if state not in registry:
            raise KeyError(f"Unknown state: {state}")
        return registry[state]

    def predict_state(self, state: str, horizon: int | None = None) -> dict[str, Any]:
        result = self.get_state_result(state)
        horizon = horizon or self.cfg.horizon
        history = result["history"].copy().sort_values("Date").reset_index(drop=True)
        model_name = result["best_model"]

        if model_name == "sarima":
            forecast_values = _sarima_forecast(result["models"]["sarima"], horizon)
        elif model_name == "prophet":
            forecast_values = _prophet_forecast(result["models"]["prophet"], history["Date"].iloc[-1], horizon, self.cfg)
        elif model_name == "xgboost":
            forecast_values = _xgb_forecast(result["models"]["xgboost"], history, horizon, self.cfg)
        else:
            forecast_values = _lstm_forecast(result["models"]["lstm"], history, horizon, self.cfg)

        next_dates = future_dates(history["Date"].iloc[-1], horizon, self.cfg.freq)
        forecast_frame = pd.DataFrame({"Date": next_dates, "Forecast": forecast_values})
        return {
            "state": state,
            "best_model": model_name,
            "metrics": result["scores"][model_name],
            "forecast": dataframe_to_json_ready(forecast_frame),
        }

    def bulk_forecast(self, horizon: int | None = None) -> list[dict[str, Any]]:
        horizon = horizon or self.cfg.horizon
        return [self.predict_state(state, horizon=horizon) for state in self.list_states()]

    def training_summary(self) -> pd.DataFrame:
        if METRICS_PATH.exists():
            return pd.read_csv(METRICS_PATH)
        self.ensure_ready()
        return pd.read_csv(METRICS_PATH)
