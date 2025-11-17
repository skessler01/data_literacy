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


def get_daily_count_per_site(data, counter_site_id, date):
    """
    Berechnet den Tageswert (Summe aller Channels) einer Zählstelle.

    Parameters:
        data (pd.DataFrame): Gesamtdaten mit 'iso_timestamp' als datetime64.
        counter_site_id (int): ID der Zählstelle.
        date (str): Datum im Format 'YYYY-MM-DD'.

    Returns:
        int: Gesamtzählwert des Tages.
        pd.DataFrame: Gefilterte Zeilen des Tages (für Debug/Analyse).
    """

    # Standort filtern
    df_site = data[data['counter_site_id'] == counter_site_id]

    # Datum in datetime umwandeln
    target_date = pd.to_datetime(date).date()

    # Tagesfilter anwenden
    df_day = df_site[df_site['iso_timestamp'].dt.date == target_date]

    # Falls keine Daten gefunden wurden
    if df_day.empty:
        print(f"Keine Daten für counter_site_id {counter_site_id} am {date}")
        return 0, df_day

    # Tageswert berechnen (über alle Channels)
    tageswert = df_day['zählstand'].sum()

    return tageswert, df_day