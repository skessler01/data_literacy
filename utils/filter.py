import pandas as pd
def filter_location_and_time(data, locations, start_date, end_date):
    # locations can be a list of strings with the names of the places
    # start_date and end_date may be string or timestamp object (as long as the string has
    # the correct format)
    # working formats are e.g. "2023-01-01T05:00:00+01:00", "2023-01-01 05:00", "2023-01-01", "2023-03",
    # "2023"
    filtered_data = data[data['standort'].isin(locations)].copy()
    filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
    filtered_data = filtered_data[(filtered_data['timestamp'] >= start_date)
                                  & (filtered_data['timestamp'] <= end_date)]
    return filtered_data