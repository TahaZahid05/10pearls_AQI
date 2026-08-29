from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZONS, TARGET_BASE_COL, TARGET_COLS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and cyclical time features from datetime timestamp."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Cyclical encodings so that model understands difference between 23:00 and 00:00 is 1 hour
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def add_weather_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute meteorological physics features including wind vector components."""
    df = df.copy()

    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        rad = np.radians(df["wind_direction_10m"])
        # u (east-west) and v (north-south) components
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        # Interaction between heat and moisture (affects smog formation and particle swelling)
        df["temp_humidity_idx"] = df["temperature_2m"] * (df["relative_humidity_2m"] / 100.0)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical lag values, rates of change, and rolling statistics."""
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # AQI Lags
    if TARGET_BASE_COL in df.columns:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"aqi_lag_{lag}h"] = df[TARGET_BASE_COL].shift(lag)

        # Rate of change / Trends
        df["aqi_change_rate_1h"] = df[TARGET_BASE_COL] - df["aqi_lag_1h"]
        df["aqi_change_rate_24h"] = df[TARGET_BASE_COL] - df["aqi_lag_24h"]

        # Rolling Statistics
        df["aqi_rolling_mean_6h"] = df[TARGET_BASE_COL].rolling(window=6, min_periods=3).mean()
        df["aqi_rolling_std_6h"] = df[TARGET_BASE_COL].rolling(window=6, min_periods=3).std().fillna(0)
        df["aqi_rolling_mean_24h"] = df[TARGET_BASE_COL].rolling(window=24, min_periods=12).mean()
        df["aqi_rolling_max_24h"] = df[TARGET_BASE_COL].rolling(window=24, min_periods=12).max()
        df["aqi_rolling_min_24h"] = df[TARGET_BASE_COL].rolling(window=24, min_periods=12).min()

    # Particulate Matter Lags (key pollutant drivers)
    for pol in ["pm2_5", "pm10"]:
        if pol in df.columns:
            df[f"{pol}_lag_1h"] = df[pol].shift(1)
            df[f"{pol}_lag_24h"] = df[pol].shift(24)

    return df


def add_target_variables(
    df: pd.DataFrame,
    horizons: List[int] = FORECAST_HORIZONS,
    target_base: str = TARGET_BASE_COL,
) -> pd.DataFrame:
    """
    Generate multi-horizon forecast targets by shifting future AQI backwards.
    target_aqi_24h: AQI 24 hours into the future (Day 1)
    target_aqi_48h: AQI 48 hours into the future (Day 2)
    target_aqi_72h: AQI 72 hours into the future (Day 3)
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    for h in horizons:
        df[f"target_aqi_{h}h"] = df[target_base].shift(-h)

    return df


def engineer_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Complete end-to-end feature engineering pipeline.

    Args:
        df: Raw combined DataFrame with air quality and weather.
        is_training: If True, adds future targets and drops rows with NaNs.
                     If False (inference), retains latest rows without requiring future targets.

    Returns:
        Processed DataFrame with all engineered features.
    """
    df = add_time_features(df)
    df = add_weather_physics(df)
    df = add_lag_and_rolling_features(df)

    if is_training:
        df = add_target_variables(df)
        # Drop initial rows with lag NaNs and trailing rows with target NaNs
        df = df.dropna().reset_index(drop=True)
    else:
        # In inference, only drop rows missing past lag history (first 48 rows if raw)
        df = df.dropna(subset=["aqi_lag_24h"]).reset_index(drop=True)

    return df


def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns the list of feature column names (X) and target column names (y)."""
    exclude_cols = ["time", "city"] + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_cols = [col for col in df.columns if col in TARGET_COLS]
    return feature_cols, target_cols
