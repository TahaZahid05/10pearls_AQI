from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZONS, TARGET_BASE_COL, TARGET_COLS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Cyclical day of week encoding
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Cyclical month encoding
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def add_weather_physics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        rad = np.radians(df["wind_direction_10m"])
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_humidity_idx"] = df["temperature_2m"] * (df["relative_humidity_2m"] / 100.0)

    return df


def add_essential_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    if TARGET_BASE_COL in df.columns:
        df["aqi_rolling_mean_24h"] = df[TARGET_BASE_COL].rolling(window=24, min_periods=12).mean()
        df["aqi_change_rate_24h"] = df[TARGET_BASE_COL] - df[TARGET_BASE_COL].shift(24)

    if "pm2_5" in df.columns:
        df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)

    return df


def add_target_variables(
    df: pd.DataFrame,
    horizons: List[int] = FORECAST_HORIZONS,
    target_base: str = TARGET_BASE_COL,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    for h in horizons:
        df[f"target_aqi_{h}h"] = df[target_base].shift(-h)

    return df


def engineer_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_weather_physics(df)
    df = add_essential_lags(df)

    if is_training:
        df = add_target_variables(df)
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.dropna(subset=["pm2_5_lag_24h"]).reset_index(drop=True)

    return df


def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude_cols = ["time", "city"] + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_cols = [col for col in df.columns if col in TARGET_COLS]
    return feature_cols, target_cols
