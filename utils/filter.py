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


def get_daily_counter_site_count(data, counter_site_id, date):
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
    daily_count = df_day['zählstand'].sum()

    return daily_count, df_day

# Tageswert eines Standorts
standort = "Stadt Tübingen"
date = "2024-01-01" # YYYY-MM-DD


def get_normalized_daily_city_count(data, standort, date):
    """
    Berechnet den durchschnittlichen Tageswert aller aktiven Counter-Sites eines Standorts.

    Parameters:
        data (pd.DataFrame): Datensatz mit Spalten 'standort', 'counter_site_id', 'iso_timestamp', 'zählstand'
                             iso_timestamp muss datetime64 sein.
        standort (str): Name des Standorts, z.B. "Stadt Tübingen"
        date (str): Datum im Format "YYYY-MM-DD"

    Returns:
        float: Durchschnittlicher Tageswert über alle aktiven Counter-Sites
        int: Anzahl aktiver Counter-Sites
        int: Summe der Daily Counts über alle aktiven Counter-Sites
    """
    # Alle Counter-Sites des Standorts
    all_sites = data[data['standort'] == standort]['counter_site_id'].unique()
    
    daily_counts = []

    # Prüfen, welche Counter am Tag aktiv sind
    target_date = pd.to_datetime(date).date()
    
    for cs_id in all_sites:
        df_site_day = data[
            (data['counter_site_id'] == cs_id) &
            (data['iso_timestamp'].dt.date == target_date)
        ]
        if not df_site_day.empty:
            daily_sum = df_site_day['zählstand'].sum()
            daily_counts.append(daily_sum)

    # Berechnung des durchschnittlichen Tageswerts
    if daily_counts:
        normalized_daily_count = sum(daily_counts) / len(daily_counts)
    else:
        normalized_daily_count = None

    return normalized_daily_count, len(daily_counts), sum(daily_counts) if daily_counts else 0