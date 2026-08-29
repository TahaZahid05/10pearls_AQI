from typing import Optional
import pandas as pd
import requests

from src.config import (
    LATITUDE,
    LONGITUDE,
    AIR_QUALITY_API_URL,
    WEATHER_FORECAST_API_URL,
    WEATHER_ARCHIVE_API_URL,
    AIR_QUALITY_VARS,
    WEATHER_VARS,
    CITY_NAME,
)


def fetch_air_quality(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    past_days: Optional[int] = None,
    forecast_days: int = 1,
) -> pd.DataFrame:
    """Fetch hourly air quality data from Open-Meteo Air Quality API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": AIR_QUALITY_VARS,
        "timezone": "auto",
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    elif past_days is not None:
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days
    else:
        params["forecast_days"] = forecast_days

    response = requests.get(AIR_QUALITY_API_URL, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data found in response: {data}")

    df = pd.DataFrame(data["hourly"])
    return df


def fetch_weather(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    past_days: Optional[int] = None,
    forecast_days: int = 1,
) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo Weather API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": WEATHER_VARS,
        "timezone": "auto",
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
        url = WEATHER_ARCHIVE_API_URL
    else:
        url = WEATHER_FORECAST_API_URL
        if past_days is not None:
            params["past_days"] = past_days
        params["forecast_days"] = forecast_days

    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data found in response: {data}")

    df = pd.DataFrame(data["hourly"])
    return df


def fetch_combined_data(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    past_days: Optional[int] = None,
    forecast_days: int = 1,
) -> pd.DataFrame:
    """Fetch and merge air quality and weather data on timestamp."""
    df_aq = fetch_air_quality(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
    )

    df_weather = fetch_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
    )

    # Merge on the shared 'time' column
    df = pd.merge(df_aq, df_weather, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = CITY_NAME
    df = df.sort_values("time").reset_index(drop=True)

    return df


print(f"Testing real-time / recent data fetch for {CITY_NAME}...")
df_test = fetch_combined_data(past_days=7, forecast_days=1)
print(f"Successfully fetched {len(df_test)} rows and {len(df_test.columns)} columns:")
print(df_test.head(3))
