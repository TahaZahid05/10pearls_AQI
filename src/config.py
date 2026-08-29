from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HISTORICAL_PARQUET = DATA_DIR / "historical_features.parquet"
HISTORICAL_CSV = DATA_DIR / "historical_features.csv"

CITY_NAME = "Karachi"
LATITUDE = 	24.8609
LONGITUDE = 66.9905
TIMEZONE = "Asia/Karachi"

AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Particulate matter (PM) is a mix of microscopic solid particles and liquid droplets suspended in the air
# Ozone
# Nitrogen dioxide
# Sulfur dioxide
# Carbon monoxide
# us_aqi = standardized air quality index
AIR_QUALITY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

# temperature_2m = air temperature at 2 meters above ground
# relative_humidity_2m = relative humidity at 2 meters above ground
# precipitation = total precipitation
# wind_speed_10m = wind speed at 10 meters above ground
# wind_direction_10m = wind direction at 10 meters above ground
# surface_pressure = atmospheric pressure at the surface
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

# Forecast Horizons in Hours (1 day, 2 days, 3 days ahead)
FORECAST_HORIZONS = [24, 48, 72]

# Target column name base
TARGET_BASE_COL = "us_aqi"
TARGET_COLS = [f"target_aqi_{h}h" for h in FORECAST_HORIZONS]
