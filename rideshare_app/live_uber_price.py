import os
import requests
import threading
import pandas as pd
import time

server_token = os.environ['STREAM_UBER_ACCESS']
uber_url = 'https://api.uber.com/'
price_url = uber_url + 'v1/estimates/price'
cities = ['denver','ny','chicago','seattle','sf']

def geolocation(city):
    """
    Output: Geolocations of the city from the downtown to the airport
    """
    city_lat_long = {
    'ny': [(40.7596515,-73.9845424),(40.6438,-73.782)],
    'seattle': [(47.6054539091,-122.334653818),(47.4510622,-122.3005362)],
    'sf': [(37.7764224703,-122.418388598),(37.61672,-122.3893932)],
    'denver': [(39.740782,-104.991153),(39.851727,-104.6738038)],
    'chicago': [(41.87768795,-87.6304131096),(41.9735512,-87.9090567)]
    }
    return city_lat_long[city][0], city_lat_long[city][1]

def multi_threading(cities):
    """
    Multithreads each city price API request
    """
    # 5 threads per city
    data = []
    threads = []
    for city in cities:
        thread = threading.Thread(target=run_thread, args=(city, data))
        threads.append(thread)
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    return data

def run_thread(city, data):
    """
    Makes request to city for price data
    """
    start, stop = geolocation(city)

    price_para = {
        'server_token': server_token,
        'start_latitude': start[0],
        'start_longitude': start[1],
        'end_latitude': stop[0],
        'end_longitude': stop[1],
    }

    r_price = requests.get(price_url, params=price_para)
    if r_price.status_code != 200:
        print 'WARNING:', r.status_code
    price_data = r_price.json()
    final_data = pd.DataFrame(price_data['prices'])
    final_data['city'] = city
    final_data['record_time'] = pd.to_datetime(time.time(), unit='s')
    final_data['avg_price_est'] = (final_data['low_estimate'] + final_data['high_estimate']) / 2.
    final_data['display_name'] = final_data['display_name'].apply(lambda x: x.lower())
    data.append(final_data[['record_time','city','display_name','avg_price_est']])

if __name__ == '__main__':
    data = multi_threading(cities)
    live_data = pd.concat(data)
