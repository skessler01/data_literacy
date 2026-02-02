import os
import pandas as pd
import requests_cache
import openmeteo_requests
import pandas as pd
import time

""" Helper functions to read in and preprocess bike count and weather data."""


def read_in_bike_data(file_path):
    """Read in bike data from local CSV or download from URLs if not present."""

    if os.path.exists(file_path):
        #print(f"File exists: {file_path}")
        bike_data_raw = pd.read_csv(file_path, low_memory=False)
        
    else:
        #print(f"File missing: {file_path}")
        url_list = []

        # URLs to data generated according to the following scheme:
        # https://mobidata-bw.de/daten/eco-counter/v2/fahrradzaehler_stundenwerten_{yyyymm}.csv.gz 

        for year in range(2013, 2026):
            for month in range(1, 13):
                yyyymm = f"{year}{month:02d}"
                url = f"https://mobidata-bw.de/daten/eco-counter/v2/fahrradzaehler_stundenwerten_{yyyymm}.csv.gz"
                
                # Check if the URL exists
                response = requests_cache.CachedSession().head(url)
                if response.status_code == 200:    
                    # Add to url_list
                    url_list.append(url)

        general_columns = pd.read_csv(url_list[1]).columns.tolist()

        # Create dataframe by concatenating all CSVs from the URLs
        full_bike_data = pd.DataFrame()
        for url in url_list:
            csv_data = pd.read_csv(url, low_memory=False)
            assert list(csv_data.columns) == general_columns, f"Spalten stimmen nicht überein in {url}"
            full_bike_data = pd.concat([full_bike_data, csv_data], ignore_index=True)

        # Save to CSV for future use
        full_bike_data.to_csv("../data/full_bike_data.csv", index=False)   
        bike_data_raw = full_bike_data.copy()
    return bike_data_raw

def cap_counter_by_time_pattern(counter_data, values_where="count", cap_const=5):
    """
    Cap outliers in counter data based on hour-of-day and day-of-year patterns.
    (quantiles and IQR calculated for each hour and day_of_year combination)

    """
    counter_data = counter_data.copy()
    
    # Extract time features
    counter_data['hour'] = counter_data['timestamp'].dt.hour
    counter_data['day_of_year'] = counter_data['timestamp'].dt.dayofyear

    # Convert to float to avoid dtype mismatch
    counter_data[values_where] = counter_data[values_where].astype(float)
    
    # Initialize capped column
    counter_data[f'{values_where}_capped'] = False
    
    # Group by hour and day_of_year and calculate IQR for each group
    for (hour, day_of_year), group_indices in counter_data.groupby(['hour', 'day_of_year']).groups.items():
        group_data = counter_data.loc[group_indices, values_where]
        
        if len(group_data) > 3:  # Need at least 4 points for meaningful quartiles
            Q1 = group_data.quantile(0.25)
            if Q1 == 0: # skip capping if Q1 is 0
                continue
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + cap_const * IQR
            
            # Mark and cap outliers
            outlier_mask = group_data > upper_bound
            counter_data.loc[group_indices[outlier_mask], f'{values_where}_capped'] = True
            counter_data.loc[group_indices[outlier_mask], values_where] = upper_bound
    
    counter_data.drop(['hour', 'day_of_year'], axis=1, inplace=True)
    return counter_data

def cap_outliers_by_time_pattern(data, values_where="count", cap_const=5):
    return data.groupby('counter_site', group_keys=False).apply(
        cap_counter_by_time_pattern, values_where=values_where, cap_const=cap_const
    )

def detect_intervals_with_missing_data(df, column='count', mode='zeros'):
    """
    Detect continuous intervals where `column` is NaN or 0.
    Returns a DataFrame with start, end, n_points.
    """
    if mode == 'zeros':
        mask = df[column] == 0
    elif mode == 'missing':
        mask = df[column].isna()
    else:
        raise ValueError("mode must be 'zeros' or 'missing'")

    grp = (mask != mask.shift()).cumsum()

    intervals = (
        df[mask]
        .assign(group=grp)
        .groupby('group')
        .agg(
            start=(column, lambda x: x.index.min()),
            end=(column, lambda x: x.index.max()),
            n_points=(column, 'size')
        )
        .reset_index(drop=True)
    )

    return intervals


def remove_long_zero_intervals(df_counter, long_zero_limit=168):
    """
    Remove intervals with more than long_zero_limit consecutive 0 counts.
    Returns cleaned df and count of removed rows.
    """
    zero_intervals = detect_intervals_with_missing_data(
        df_counter.set_index('timestamp'),
        column='count',
        mode='zeros'
    )

    intervals_to_remove = zero_intervals[
        zero_intervals['n_points'] > long_zero_limit
    ]

    exclude_mask = pd.Series(False, index=df_counter.index)

    for _, row in intervals_to_remove.iterrows():
        exclude_mask |= (
            (df_counter['timestamp'] >= row['start']) &
            (df_counter['timestamp'] <= row['end'])
        )

    removed_count = exclude_mask.sum()
    df_cleaned = df_counter.loc[~exclude_mask].copy()

    return df_cleaned, removed_count


def interpolate_short_gaps(df, missing_intervals, threshold):
    """
    Interpolate gaps with <= threshold missing points.
    Returns interpolated df and number of interpolated points.
    """
    to_interpolate = missing_intervals[
        missing_intervals['n_points'] <= threshold
    ]

    interpolate_mask = pd.Series(False, index=df.index)

    for _, row in to_interpolate.iterrows():
        interpolate_mask |= (
            (df.index >= row['start']) &
            (df.index <= row['end'])
        )

    df.loc[interpolate_mask, 'count'] = (
        df['count'].interpolate(method='time')
    )

    return df, interpolate_mask.sum()


def split_long_gaps(df, long_gaps, counter_col='counter_site'):
    """
    Split dataframe into multiple counters around long gaps.
    """
    if long_gaps.empty:
        return df

    counter_base = df[counter_col].iloc[0]
    segments = []
    last_end = df.index.min()

    for _, row in long_gaps.iterrows():
        segment = df.loc[
            last_end : row['start'] - pd.Timedelta(hours=1)
        ].copy()
        if not segment.empty:
            segments.append(segment)
        last_end = row['end'] + pd.Timedelta(hours=1)

    tail = df.loc[last_end:].copy()
    if not tail.empty:
        segments.append(tail)

    processed = []
    for i, seg in enumerate(segments):
        seg = seg.copy()
        seg[counter_col] = (
            counter_base if i == 0 else f"{counter_base}_{i+1}"
        )
        processed.append(seg)

    return pd.concat(processed).sort_index()

def handle_missing_data(data, long_zero_limit, interpolation_limit):
    """Handle missing data in bike counter dataset per counter site.

    Parameters:
    data - pd.DataFrame containing bike counter data 
    long_zero_limit - int, maximum length of zero-count intervals to keep (in hours)
    interpolation_limit - int, maximum length of missing data intervals to interpolate (in hours)

    Returns:
    bike_data_final - pd.DataFrame with missing data handled
    summary_df - pd.DataFrame summarizing the processing per counter site
    """

    all_counters_processed = []
    summary_list = []

    for city in data['city'].unique():
        print(f"\nProcessing city: {city}\n{'='*80}")
        df_city = data[data['city'] == city]

        for counter in df_city['counter_site'].unique():
            print(f"Processing counter: {counter}")

            df_counter = df_city[
                df_city['counter_site'] == counter
            ].copy()

            df_counter['timestamp'] = pd.to_datetime(
                df_counter['timestamp'], utc=True
            )
            df_counter['count'] = pd.to_numeric(
                df_counter['count'], errors='coerce'
            )

            # -------------------------------------------------
            # Remove long zero-count intervals
            # -------------------------------------------------
            df_counter, removed_count = remove_long_zero_intervals(
                df_counter, long_zero_limit
            )

            # -------------------------------------------------
            # Reindex to hourly frequency
            # -------------------------------------------------
            df_counter = df_counter.set_index('timestamp').sort_index()

            full_index = pd.date_range(
                df_counter.index.min(),
                df_counter.index.max(),
                freq='H',
                tz='UTC'
            )

            df_counter = df_counter.reindex(full_index)
            df_counter.index.name = 'timestamp'

            # -------------------------------------------------
            # Detect missing intervals
            # -------------------------------------------------
            missing_intervals = detect_intervals_with_missing_data(
                df_counter,
                column='count',
                mode='missing'
            )

            total_missing = missing_intervals['n_points'].sum()

            # -------------------------------------------------
            # Interpolate short gaps
            # -------------------------------------------------
            df_counter, interpolated_count = interpolate_short_gaps(
                df_counter,
                missing_intervals,
                interpolation_limit
            )

            # -------------------------------------------------
            # Fill metadata
            # -------------------------------------------------
            meta_cols = ['city', 'counter_site', 'longitude', 'latitude']
            df_counter[meta_cols] = df_counter[meta_cols].ffill().bfill()

            # -------------------------------------------------
            # Split long gaps
            # -------------------------------------------------
            long_gaps = missing_intervals[
                missing_intervals['n_points'] > interpolation_limit
            ]

            df_counter = split_long_gaps(df_counter, long_gaps)

            # -------------------------------------------------
            # Final validation + collection
            # -------------------------------------------------
            for site, g in df_counter.groupby('counter_site'):
                if g['count'].isna().any():
                    raise ValueError(
                        f"NaNs remain after processing counter {site}"
                    )

                all_counters_processed.append(g)
                summary_list.append({
                    'city': city,
                    'counter_site': site,
                    'total_missing': total_missing,
                    'interpolated': interpolated_count,
                    'removed zeros': removed_count
                })


    # =====================================================
    # Final outputs
    # =====================================================

    bike_data_final = (
        pd.concat(all_counters_processed)
        .sort_index()
        .reset_index()
        .rename(columns={'index': 'timestamp'})
    )

    summary_df = pd.DataFrame(summary_list)

    return bike_data_final, summary_df

# Downloads hourly weatherdata within the given time frame (start_data to end_date) for given coordinates
# Allows downloading from historic-forecast if forecast=true or from a observed weather archive if forecast=False
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
    weather_data = hourly_dataframe.copy()

    if forecast:
        weather_data = weather_data.rename(columns={
            "temperature_2m": "forecast_temperature_2m",
            "apparent_temperature": "forecast_apparent_temperature",
            "rain": "forecast_rain",
            "snowfall": "forecast_snowfall",
        })

    return weather_data

# Downloads both, forecast data and observed data and returns it as a single dataframe
def get_full_weather_data(start_date, end_date, longitude, latitude):
    # --- 1. Actual weather: full range ---
    actual_weather_data = get_weather_data(
        start_date=start_date,
        end_date=end_date,
        longitude=longitude,
        latitude=latitude,
        forecast=False
    )

    # --- 2. Forecast weather: clipped range ---
    forecast_start, forecast_end = clip_forecast_dates(start_date, end_date)

    # Only fetch if there is any overlap
    if pd.to_datetime(forecast_end) >= pd.to_datetime(forecast_start):
        forecast_weather_data = get_weather_data(
            start_date=forecast_start,
            end_date=forecast_end,
            longitude=longitude,
            latitude=latitude,
            forecast=True
        )
    else:
        # No forecast data available; create empty frame with correct columns
        forecast_weather_data = pd.DataFrame({
            "date": [],
            "forecast_temperature_2m": [],
            "forecast_apparent_temperature": [],
            "forecast_rain": [],
            "forecast_snowfall": []
        })

    # --- 3. Merge with LEFT JOIN to preserve historic rows ---
    actual_weather_data["date"] = pd.to_datetime(actual_weather_data["date"], utc=True)
    forecast_weather_data["date"] = pd.to_datetime(forecast_weather_data["date"], utc=True)

    merged = pd.merge(actual_weather_data, forecast_weather_data, on="date", how="left")
    return merged
from datetime import date

FORECAST_MIN_DATE = date(2016, 1, 1)
FORECAST_MAX_DATE = date(2026, 1, 18)

# helper method to ensure data validity for forecast data
def clip_forecast_dates(start_date: str, end_date: str):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    # Clip
    start_clipped = max(start, FORECAST_MIN_DATE)
    end_clipped   = min(end, FORECAST_MAX_DATE)

    # Return ISO format
    return start_clipped.isoformat(), end_clipped.isoformat()

# helper method to merge bikedata with weatherdata in one line
def merge_bike_weather(bike_data, weather_data):
    bike_data = bike_data.copy()
    weather_data = weather_data.copy()
    bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'], utc=True)
    weather_data['date'] = pd.to_datetime(weather_data['date'], utc=True)
    weather_data = weather_data.rename(columns={'date': 'timestamp'})  # rename column for easier merge
    fused_data = pd.merge(bike_data, weather_data, on=['city','timestamp'], how='inner')  # merge
    return fused_data


# takes the coordinates from a dataset grouped by city and downloads the weather data
# corresponding to the time frame of the data. Saves weather data and
# @returns the corresponding weather data
def download_corresponding_weather_data(data):
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

        time.sleep(16)  # <-- sleep between requests to not exceed the request limit

    weather_data = pd.concat(weather_frames, ignore_index=True)
    weather_data.to_csv("../data/weather_data.csv", index=False)
    return weather_data