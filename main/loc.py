import pandas as pd
import googlemaps
from googlemaps.geocoding import geocode, reverse_geocode


def extract_address(response):
    return response[0]['formatted_address']


def extract_location(response):
    return response[0]['geometry']['location']


def find_locs(df: pd.DataFrame):
    client = googlemaps.Client('AIzaSyBBJyrkuTdzQGG30_Dc4kboECFjP6bM43I')

    # ЗАДАНИЕ - ЗАПОЛНИТЬ НЕДОСТАЮЩИЕ ЛОКАЦИИ

    no_loc = 'tweet_location.isnull().values & tweet_coord.notnull().values'
    df_noloc = df.query(no_loc).loc[:, ['tweet_coord', 'tweet_location']]
    addresses = []
    for row in range(len(df_noloc)):
        coords = df_noloc.iloc[row, 0].strip('[]')
        response = reverse_geocode(client, coords)
        if response:
            address = extract_address(response)
            addresses.append(address)
        else:
            addresses.append('Unknown location')
    df.loc[df_noloc.index.values, 'tweet_location'] = addresses

    # ЗАДАНИЕ - ЗАПОЛНИТЬ ОТСУТСТВУЮЩИЕ КООРДИНАТЫ

    no_coord = 'tweet_coord.isnull().values & tweet_location.notnull().values'
    df_nocoord = df.query(no_coord).loc[:, ['tweet_coord', 'tweet_location']]
    locations = []
    for row in range(len(df_nocoord)):
        loc = df_nocoord.iloc[row, 1]
        response = geocode(client, loc)
        if response:
            coords = extract_location(response)
            lat, long = coords['lat'], coords['lng']
            locations.append((lat, long))
        else:
            locations.append('Unknown coordinates')
    df.loc[df_nocoord.index.values, 'tweet_coord'] = locations
