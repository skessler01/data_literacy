import openmeteo_requests
import pandas as pd
import requests_cache
import time


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
	    "hourly": ["temperature_2m", "apparent_temperature", "rain", "snowfall"],
        "timezone": "Europe/Berlin",
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process hourly data. The order of variables needs to be the same as requested.
    response = responses[0]
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m, "apparent_temperature": hourly_apparent_temperature, "rain": hourly_rain, "snowfall": hourly_snowfall}

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    weather_data = hourly_dataframe[['date',"temperature_2m", "apparent_temperature", 'rain', 'snowfall']].copy()

    if forecast:
        weather_data = weather_data.rename(columns={'rain': 'forecast_rain'})
        weather_data = weather_data.rename(columns={'snowfall': 'forecast_snowfall'})

    return weather_data


def get_full_weather_data(start_date, end_date, longitude, latitude):
    forecast_weather_data = get_weather_data(start_date, end_date, longitude, latitude, True)
    actual_weather_data = get_weather_data(start_date, end_date, longitude, latitude, False)
    forecast_weather_data['date'] = pd.to_datetime(forecast_weather_data['date'], utc=True)
    actual_weather_data['date'] = pd.to_datetime(actual_weather_data['date'], utc=True)
    merged_data = pd.merge(actual_weather_data, forecast_weather_data, on='date', how='inner')
    return merged_data

def merge_bike_weather(bike_data, weather_data):
    bike_data = bike_data.copy()
    weather_data = weather_data.copy()
    bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'], utc=True)
    weather_data['date'] = pd.to_datetime(weather_data['date'], utc=True)
    weather_data = weather_data.rename(columns={'date': 'timestamp'})  # rename column for easier merge
    fused_data = pd.merge(bike_data, weather_data, on=['city','timestamp'], how='inner')  # merge
    return fused_data

def download_corresponding_weather_data(data):
    # takes the coordinates from a dataset grouped by city and downloads the weather data
    # corresponding to the time frame of the data. Saves weather data and
    # @returns the corresponding weather data
    city_coords = (
        data[["city", "city_lat", "city_lon"]]
        .drop_duplicates(subset="city")
    )

    start_date = pd.to_datetime(data["timestamp"], utc=True).min().date().isoformat()
    end_date = pd.to_datetime(data["timestamp"], utc=True).max().date().isoformat()

    print("Global start:", start_date)
    print("Global end:", end_date)

    weather_frames = []

    for _, row in city_coords.iterrows():
        city = row["city"]
        lat = row["city_lat"]
        lon = row["city_lon"]

        print(f"Fetching weather for {city}: ({lat}, {lon})")

        df_weather = get_weather_data(
            start_date=start_date,
            end_date=end_date,
            longitude=lon,
            latitude=lat,
            forecast=False
        )

        df_weather["city"] = city
        weather_frames.append(df_weather)

        time.sleep(16)  # <-- important: sleep between requests

    weather_data = pd.concat(weather_frames, ignore_index=True)
    weather_data.to_csv("../data/weather_data.csv", index=False)
    return weather_data