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
    print(f"Fetching historical data for {CITY_NAME} ({start_date} to {end_date})...")
    df_raw = fetch_combined_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"Fetched {len(df_raw)} raw records")

    print("Computing features...")
    df_features = engineer_features(df_raw, is_training=True)
    print(f"Features ready: {len(df_features)} rows, {len(df_features.columns)} columns")

    feature_cols, target_cols = get_feature_and_target_columns(df_features)
    print(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")

    if save_parquet:
        df_features.to_parquet(HISTORICAL_PARQUET, index=False)
        print(f"Saved to {HISTORICAL_PARQUET}")

    if save_csv:
        df_features.to_csv(HISTORICAL_CSV, index=False)
        print(f"Saved to {HISTORICAL_CSV}")

    return df_features


if __name__ == "__main__":
    run_backfill()
