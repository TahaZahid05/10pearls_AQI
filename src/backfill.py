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
    end_date: str | None = None,
    save_parquet: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

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


def update_features_incremental() -> pd.DataFrame:
    print(f"Fetching recent real-time observations for {CITY_NAME}...")
    df_recent_raw = fetch_combined_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        past_days=7,
        forecast_days=1,
    )
    df_recent_features = engineer_features(df_recent_raw, is_training=True)

    if HISTORICAL_PARQUET.exists():
        df_hist = pd.read_parquet(HISTORICAL_PARQUET)
        prev_count = len(df_hist)
        df_combined = pd.concat([df_hist, df_recent_features], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["time"], keep="last")
        df_combined = df_combined.sort_values("time").reset_index(drop=True)
        print(f"Merged features: {prev_count} -> {len(df_combined)} rows (+{len(df_combined) - prev_count} new)")
    else:
        df_combined = df_recent_features

    df_combined.to_parquet(HISTORICAL_PARQUET, index=False)
    print(f"Saved updated dataset to {HISTORICAL_PARQUET}")
    return df_combined


if __name__ == "__main__":
    if HISTORICAL_PARQUET.exists():
        update_features_incremental()
    else:
        run_backfill()
