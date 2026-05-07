from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import holidays
import numpy as np
import pandas as pd

from config import DATA_PATH, ForecastConfig


WEEKDAY_MAP = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def load_sales_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["State"] = df["State"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def get_state_names(df: pd.DataFrame) -> list[str]:
    return sorted(df["State"].dropna().unique().tolist())


def _holiday_flag(dates: pd.DatetimeIndex) -> pd.Series:
    years = range(dates.min().year, dates.max().year + 1)
    us_holidays = holidays.US(years=years)
    return pd.Series([1 if d.date() in us_holidays else 0 for d in dates], index=dates)


def make_weekly_frame(state_df: pd.DataFrame, cfg: ForecastConfig = ForecastConfig()) -> pd.DataFrame:
    if state_df.empty:
        raise ValueError("State dataframe is empty")

    state_df = state_df.sort_values("Date").copy()
    start = state_df["Date"].min()
    end = state_df["Date"].max()
    full_index = pd.date_range(start=start, end=end, freq=cfg.freq)

    frame = state_df.set_index("Date").reindex(full_index)
    frame.index.name = "Date"

    frame["State"] = state_df["State"].iloc[0]
    frame["Category"] = state_df["Category"].mode(dropna=True).iloc[0] if "Category" in state_df.columns else "Unknown"

    frame["Total"] = frame["Total"].interpolate(method="time", limit_direction="both")
    frame["Total"] = frame["Total"].ffill().bfill()

    frame["holiday_flag"] = _holiday_flag(frame.index).values
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["quarter"] = frame.index.quarter
    frame["weekofyear"] = frame.index.isocalendar().week.astype(int)
    frame["trend_idx"] = np.arange(len(frame), dtype=int)
    frame["is_month_start"] = frame.index.is_month_start.astype(int)
    frame["is_month_end"] = frame.index.is_month_end.astype(int)

    frame = frame.reset_index()
    frame["weekday_name"] = frame["Date"].dt.dayofweek.map(WEEKDAY_MAP)
    return frame


def add_lag_and_rolling_features(frame: pd.DataFrame, cfg: ForecastConfig = ForecastConfig()) -> pd.DataFrame:
    out = frame.copy().sort_values("Date").reset_index(drop=True)
    target = out["Total"]

    for lag in cfg.lag_features:
        out[f"lag_{lag}"] = target.shift(lag)

    for window in cfg.rolling_windows:
        out[f"roll_mean_{window}"] = target.shift(1).rolling(window=window).mean()
        out[f"roll_std_{window}"] = target.shift(1).rolling(window=window).std().fillna(0.0)

    out["ewm_4"] = target.shift(1).ewm(span=4, adjust=False).mean()
    out["ewm_8"] = target.shift(1).ewm(span=8, adjust=False).mean()

    return out


def make_supervised_frame(frame: pd.DataFrame, cfg: ForecastConfig = ForecastConfig()) -> pd.DataFrame:
    out = add_lag_and_rolling_features(frame, cfg)
    out = out.dropna().reset_index(drop=True)
    return out


def train_validation_split(frame: pd.DataFrame, validation_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) <= validation_size:
        raise ValueError("Not enough rows to split into train and validation sets")
    train = frame.iloc[:-validation_size].copy().reset_index(drop=True)
    valid = frame.iloc[-validation_size:].copy().reset_index(drop=True)
    return train, valid


def future_dates(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)


def dataframe_to_json_ready(frame: pd.DataFrame) -> list[dict]:
    records = frame.copy()
    for column in records.columns:
        if pd.api.types.is_datetime64_any_dtype(records[column]):
            records[column] = records[column].dt.strftime("%Y-%m-%d")
    return records.to_dict(orient="records")


def json_ready_to_dataframe(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
    return frame
