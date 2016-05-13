__author__ = 'timhou'
import requests
# from geopy.geocoders import Nominatim
from collections import defaultdict
from pymongo import MongoClient
import time
import json
import os
import threading

server_token = os.environ['UBER_ACCESS_KEY']
client = MongoClient()
db = client['apidata']
collection = db['uberapi']

uber_url = 'https://api.uber.com/'
# products endpoint returns information about Uber products offered at a given location
product_url = uber_url + 'v1/products'
# time estimate endpoint returns ETAs for all products offered at a given location (expressed in seconds)
time_estimate_url = uber_url + 'v1/estimates/time'
# price estimates endpoint returns an estimated price range for each product offered at a given location
price_estimate_url = uber_url + 'v1/estimates/price'

class UberAPI(object):
    """
    Rate Limit: 1000 request per hour
    Resolution: 3 requests per minute
    Requests: List of products, price estimates, time estimates
        - 5 cities: 900 requests per hour (1 route)
        - 6 cities: 1080 requests per hour (1 route)
    Route: Downtown to Airport
    Cities: Denver, New York, Chicago, San Francisco, Seattle (900 requests per hour)
    """

    def __init__(self):
        """
        Initialize addresses and geolocations for cities of interest
        """
        self._get_geolocation()
        self._get_cities_addresses()

    def _get_cities_addresses(self):
        """
        Output: Stores all cities addresses into a dictionary
        """
        self.cities = ['denver','ny','chicago','seattle','sf'] #,'austin','boston']
        '''
        Downtown -> Airport
        '''
        self.denver = ('301 14th St, Denver, CO 80202','8500 Pena Blvd, Denver, CO 80249')
        self.ny = ('712 7th Ave, New York, NY 10036','New York, NY 11430')
        self.chicago = ('77 W Jackson Blvd, Chicago, IL 60604',"A10, O'Hare Commercial Arrivals, Chicago, 60666")
        self.seattle = ('248 Madison St, Seattle, WA 98104','9075 Airport Expy, SeaTac, WA 98188')
        self.sf = ('39 Fell St, San Francisco, CA 94102','SFO International Terminal Main Hall, San Mateo County, CA, 94128')
        self.austin = ('300 W 10th St, Austin, TX 78701','3600 Presidential Boulevard, Austin, TX 78719')
        self.boston = ('53 Hawley St, Boston, MA 02110','1 Harborside Dr, Boston, MA 02128')

        self.addresses = [self.denver] + [self.ny] + [self.chicago] + [self.seattle] + [self.sf] + [self.austin] + [self.boston]

        # store all of the addresses and cities in dictionary
        self.all_addresses = defaultdict(list)
        for i,c in enumerate(self.cities):
            self.all_addresses[c] = self.addresses[i]


    def _get_geolocation(self):
        """
        Get geolocations of the city from the downtown to the airport
        """
        self.city_lat_long = {
        'ny': [(40.7596515,-73.9845424),(40.6438,-73.782)],
        'seattle': [(47.6054539091,-122.334653818),(47.4510622,-122.3005362)],
        'sf': [(37.7764224703,-122.418388598),(37.61672,-122.3893932)],
        'denver': [(39.740782,-104.991153),(39.851727,-104.6738038)],
        'chicago': [(41.87768795,-87.6304131096),(41.9735512,-87.9090567)]
        }

    def run(self):
        """
        Output: Gets all city addresses and then makes requests to Uber API to obtain product, time and price estimates
        """
        self.multi_threading(server_token, product_url, time_estimate_url, price_estimate_url, collection)

        print 'done at {0}. count of docs in mongo: {1}.'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), collection.count())

        client.close()

    def multi_threading(self, server_token, product_url, time_url, price_url, collection):
        """
        Input: Dictionary of city addresses, all API request urls from Uber, secret Uber token, and storage collection for MongoDB
        Output: Starts multithreading process for each city request
        """
        # 5 threads per city
        threads = []
        for city,adds in self.all_addresses.items():
            thread = threading.Thread(target=self.run_thread, args=(city, adds, server_token, product_url, time_url, price_url, collection))
            threads.append(thread)
        for thread in threads: thread.start()
        for thread in threads: thread.join()

    def run_thread(self, city, addresses, server_token, product_url, time_url, price_url, collection):
        """
        Input: Single city address, all API request urls from Uber, secret Uber token, and storage collection for MongoDB
        Output: obtains start and stop latitude/longitude coordinates, then stores them for the API parameter, then makes a request, and stores that document into MongoDB
        """
        self.start, self.stop = self.city_lat_long[city][0], self.city_lat_long[city][1]
        #print city, start[0], start[1], stop[0], stop[1]

        # setup parameters for each request from uber for each city
        self._get_parameters(server_token)

        # make requests to uber to obtain json documents of product, time, and price data for each city
        self._make_requests(product_url, time_url, price_url)

        self.data = {
            'city': city,
            'start_address': addresses[0],
            'start_latitude': self.start[0],
            'start_longitude': self.start[1],
            'stop_address': addresses[1],
            'stop_latitude': self.stop[0],
            'stop_longitude': self.stop[1],
            'record_time': time.time(),
            'products:': self.product_data.json()['products'],
            'times': self.time_data.json()['times'],
            'prices': self.price_data.json()['prices']
        }

        collection.insert_one(self.data)

    def _get_parameters(self, server_token):
        """
        Input: start/stop latitude and longitude of city
        Output: dictionaries of parameters for uber api request
        """
        self.product_para = {
            'server_token': server_token,
            'latitude': self.start[0],
            'longitude': self.start[1],
        }
        self.time_estimate_para = {
            'server_token': server_token,
            'start_latitude': self.start[0],
            'start_longitude': self.start[1],
        }
        self.price_estimate_para = {
            'server_token': server_token,
            'start_latitude': self.start[0],
            'start_longitude': self.start[1],
            'end_latitude': self.stop[0],
            'end_longitude': self.stop[1],
        }

    def _make_requests(self, product_url, time_url, price_url):
        """
        Input: parameters and URL for all API requests
        Output: Sends API requests to Uber to obtain all json documents for each request
        """
        # check to make sure we are only recieving status code 200
        self.product_data = requests.get(product_url, params=self.product_para)
        self.time_data = requests.get(time_url, params=self.time_estimate_para)
        self.price_data = requests.get(price_url, params=self.price_estimate_para)
        for r in [self.product_data,self.time_data,self.price_data]:
            if r.status_code != 200:
                print 'WARNING:', r.status_code

if __name__ == '__main__':
    # with open('./.uber.json') as f:
    #     key = json.loads(f.read())
    #     server_token = key['token']
    uar = UberAPI()
    uar.run()
