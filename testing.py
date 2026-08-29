import requests
import pandas as pd

latitude = 24.8607
longitude = 67.0011

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"],
    "past_days": 30,
    "forecast_days": 1
}
aq_response = requests.get(aq_url, params=aq_params).json()


weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "surface_pressure"],
    "past_days": 30,
    "forecast_days": 1
}
weather_response = requests.get(weather_url, params=weather_params).json()

df_aq = pd.DataFrame(aq_response["hourly"])
df_weather = pd.DataFrame(weather_response["hourly"])
df = pd.merge(df_aq, df_weather, on="time")

print(df.head())
