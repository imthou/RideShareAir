import requests
import os
import pandas as pd
import json
from pymongo import MongoClient
import time

client = MongoClient()
db = client['forecastdata']
collection = db['forecastapi']
forecast_key = os.environ['FORECAST_KEY']

class ForecastAPIRequest(object):
    """
    Obtains weather data specific to the cities for the ride sharing data from Forecast.io
    """
    def __init__(self, start_date='2016-02-15', end_date='2016-03-30'):
        self.cities = ['denver','ny','chicago','seattle','sf']
        self.dates = pd.date_range(start=start_date, end=end_date)

    def run(self):
        """
        Runs request for weather data
        """
        self.request_weather_data()
        client.close()

    def request_weather_data(self):
        """
        Output: Dictionary

        Stores weather data in MongoDB
        """
        for city in self.cities:
            start, _ = self.geolocation(city)
            for date in self.dates:
                c_date = date.isoformat()
                url = "https://api.forecast.io/forecast/{}/{},{},{}".format(forecast_key,start[0],start[1],c_date)
                r = requests.get(url)
                d = json.loads(r.text)
                if r.status_code != 200:
                    print "Problem with request of data"
                d['city'] = city
                collection.insert_one(d)
                print 'done at {0}. count of docs in mongo: {1}.'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), collection.count())


    def geolocation(self, city):
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

if __name__ == '__main__':
    far = ForecastAPIRequest(start_date='2016-02-15', end_date='2016-03-30')
    far.run()
