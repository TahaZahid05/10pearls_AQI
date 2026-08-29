from datetime import datetime, timedelta
import pandas as pd

from src.config import (
    CITY_NAME,
    LATITUDE,
    LONGITUDE,
    HISTORICAL_PARQUET,
    HISTORICAL_CSV,
)
from src.data_fetcher import fetch_combined_data
from src.feature_engineering import (
    engineer_features,
    get_feature_and_target_columns,
)


def run_backfill(
    start_date: str = "2025-08-01",
    end_date: str = "2026-08-28",
    save_parquet: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Backfills historical data for the given date range, transforms it, and saves to disk.
    """
    print(f"==================================================")
    print(f"Starting Historical Backfill for {CITY_NAME}")
    print(f"Coordinates: ({LATITUDE}, {LONGITUDE})")
    print(f"Date Range: {start_date} -> {end_date}")
    print(f"==================================================")

    # 1. Fetch raw data from Open-Meteo in a single query
    print("Fetching air quality and weather data from Open-Meteo.")
    df_raw = fetch_combined_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"Raw Data Fetched: {len(df_raw)} hourly records across {len(df_raw.columns)} columns.")

    # 2. Engineer features & targets
    print("Applying feature engineering (time, cyclical, wind vectors, lags, rolling stats, 3-day targets)...")
    df_features = engineer_features(df_raw, is_training=True)
    print(f"Feature Engineering Complete: {len(df_features)} clean rows, {len(df_features.columns)} total columns.")

    feature_cols, target_cols = get_feature_and_target_columns(df_features)
    print(f"Features count (X): {len(feature_cols)}")
    print(f"Targets count (y): {len(target_cols)} ({', '.join(target_cols)})")

    # 3. Save to disk
    if save_parquet:
        df_features.to_parquet(HISTORICAL_PARQUET, index=False)
        print(f"Saved Parquet dataset to: {HISTORICAL_PARQUET}")

    if save_csv:
        df_features.to_csv(HISTORICAL_CSV, index=False)
        print(f"Saved CSV dataset to: {HISTORICAL_CSV}")

    # 4. Summary metrics
    print("\n--- Summary Statistics of Target Variables ---")
    print(df_features[target_cols].describe().round(2))

    return df_features


df = run_backfill()
