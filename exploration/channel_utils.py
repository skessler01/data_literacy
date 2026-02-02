import pandas as pd

#data = pd.read_csv("..alle_fahrradzaehler_daten.csv")


# Funktionen können in Jupyter Notebooks mit helper_functions.<Funktionsname> aufgerufen werden, import helper_functions vorausgesetzt 

def get_all_locations(df: pd.DataFrame) -> pd.DataFrame:
    df_locations = df[['domain_name', 'domain_id', 'counter_site', 'counter_site_id', 'latitude', 'longitude']]
    # Entfernen von doppelten Einträgen
    df_locations = df_locations.drop_duplicates()
    return df_locations

# Gibt alle Standorte (Locations) für eine bestimmte Domain zurück, domains sind die Städte/Kommunen
def locations_in_domain(domain_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df_locations = df[df['domain_name'] == domain_name][['domain_name', 'domain_id', 'counter_site', 'counter_site_id', 'latitude', 'longitude']]
    # Entfernen von doppelten Einträgen
    df_locations = df_locations.drop_duplicates()
    return df_locations

# Gibt alle einzigartigen Kanalnamen (Channels) zurück
def get_all_channels(df: pd.DataFrame) -> list:
    return df['channel_name'].unique().tolist()

# Gibt alle Kanalnamen zurück, die weder Autos noch Fußgänger zählen
def get_normal_channels(df: pd.DataFrame) -> list:
    auto_channels = [col for col in df['channel_name'].unique() if 'Auto' in col or 'auto' in col or 'PKW' in col or 'pkw' in col or 'Car' in col or 'car' in col or 'Vehicle' in col or 'vehicle' in col or 'Fahrzeug' in col or 'fahrzeug' in col]
    foot_channels = [col for col in df['channel_name'].unique() if 'Fußgänger' in col or 'Fussgänger' in col or 'fußgänger' in col or 'fussgänger' in col or 'Person' in col or 'person' in col or 'pedestrian' in col or 'Pedestrian' in col]
    strange_channels = [col for col in df['channel_name'].unique() if '<' in col or '>' in col]
    normal_channels = [col for col in df['channel_name'].unique() if col not in auto_channels and col not in foot_channels and col not in strange_channels]
    return normal_channels  

# Filtert die Daten nach einer Liste von Channels
def filter_data_by_channels(channel_names: list, df: pd.DataFrame) -> pd.DataFrame:
    return df[df['channel_name'].isin(channel_names)]

