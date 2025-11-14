import openmeteo_requests
import pandas as pd
import requests_cache


def get_weather_data(start_date, end_date, longitude, latitude, forecast):

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    openmeteo = openmeteo_requests.Client(session=cache_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    if forecast:
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ["temperature_2m", "rain", "snowfall"],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process hourly data. The order of variables needs to be the same as requested.
    response = responses[0]
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_rain = hourly.Variables(1).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m, "rain": hourly_rain, "snowfall": hourly_snowfall}

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    weather_data = hourly_dataframe[['date', 'rain', 'snowfall']].copy()
    # Convert rain and snowfall to True/False
    weather_data['rain'] = weather_data['rain'] > 0
    weather_data['snowfall'] = weather_data['snowfall'] > 0

    if forecast:
        weather_data = weather_data.rename(columns={'rain': 'forecast_rain'})
        weather_data = weather_data.rename(columns={'snowfall': 'forecast_snowfall'})

    return weather_data

def fuse_bike_weather(bike_data, weather_data, forecast_weather_data):
    bike_data = bike_data.copy()
    weather_data = weather_data.copy()
    forecast_weather_data = forecast_weather_data.copy()

    bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'], utc=True)
    weather_data['date'] = pd.to_datetime(weather_data['date'], utc=True)
    forecast_weather_data['date'] = pd.to_datetime(forecast_weather_data['date'], utc=True)

    weather_data = weather_data.rename(columns={'date': 'timestamp'})  # rename column for easier merge
    forecast_weather_data = forecast_weather_data.rename(columns={'date': 'timestamp'})

    fused_data = pd.merge(bike_data, weather_data, on='timestamp', how='inner')  # merge
    fused_data = pd.merge(fused_data, forecast_weather_data, on='timestamp', how='inner')

    return fused_data