import requests
import os
import urllib
import urllib2
import json
from pymongo import MongoClient
import threading
from collections import defaultdict
import time

"""
Replicating this bash command in urllib2 to obtain lyft auth_token

curl -X POST -H "Content-Type: application/json" \
 --user "<client_id>:<client_secret>" \
 -d '{"grant_type": "client_credentials", "scope": "public"}' \
 'https://api.lyft.com/oauth/token'

curl --include -X GET -H 'Authorization: Bearer <bearer_token>' \
     'https://api.lyft.com/v1/ridetypes?lat=37.7833&lng=-122.4167'
"""

client = MongoClient()
db = client['lyftdata']
collection = db['lyftapi']
lyft_client_id = os.environ['LYFT_CLIENT_ID']
lyft_client_secret = os.environ['LYFT_CLIENT_SECRET']

class LyftAPIRequest(object):
    """
    Obtains real-time Lyft product availability
    """
    def __init__(self):
        """
        Initialize with lyft client information
        """
        self.lyft_client_id = lyft_client_id
        self.lyft_client_secret = lyft_client_secret
        self.url = 'https://api.lyft.com/v1/'

    def run(self):
        """
        Output: Dictionary

        Obtains a dictionary of available lyft products
        """
        if not os.path.isfile('.lyft_token.json'):
            self.access_token = self.get_auth_token()
        else:
            self.access_token = self.read_token()
        self.test_token()
        self.multi_threading()

        print 'done at {0}. count of docs in mongo: {1}.'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), collection.count())

        client.close()

    def test_token(self):
        """
        Output: String

        Test if the token has expired (401 error), if so, renew it
        """
        start, stop = self.geolocation('denver')
        avail_data = urllib.urlencode({"lat": start[0], "lng": start[1]})
        req = urllib2.Request(url=self.url + 'ridetypes' + '?' + avail_data, headers={"Authorization": "Bearer {}".format(self.access_token)})
        r = urllib2.urlopen(req)
        if r.code == 401:
            print "Token has expired, renewing token"
            self.access_token = self.get_auth_token()

    def read_token(self):
        """
        Output: String

        Reads in auth token from json file
        """
        with open('.lyft_token.json', 'r') as infile:
            self.token_d = json.load(infile)
        return self.token_d['access_token']

    def get_auth_token(self):
        """
        Output: String

        Obtains access token
        """
        data = '{"grant_type": "client_credentials", "scope": "public"}'
        url = 'https://api.lyft.com/oauth/token'
        req = urllib2.Request(url=url, data=data, headers={"Authorization": self.basic_authorization(lyft_client_id, lyft_client_secret), 'Content-Type': 'application/json'})
        r = urllib2.urlopen(req)
        if r.code != 200:
            print "error request auth token: {}".format(r.read())
        else:
            self.token_d = json.loads(r.read())
            access_token = self.token_d['access_token']
            r.close()
        with open('lyft_token.json', 'w') as outfile:
            json.dump(self.token_d, outfile)
        return access_token

    def basic_authorization(self, user, password):
        """
        Output: String

        Returns client and client password encoded in base64
        """
        s = user + ":" + password
        return "Basic " + s.encode("base64").rstrip()

    def multi_threading(self):
        """
        Input: Dictionary of city addresses, all API request urls from Lyft, secret Uber token, and storage collection for MongoDB

        Output: Starts multithreading process for each city request
        """
        # 5 threads per city
        cities_dct = self.cities_addresses()
        threads = []
        for city,adds in cities_dct.items():
            thread = threading.Thread(target=self.get_response_json_object, args=(city, adds))
            threads.append(thread)
        for thread in threads: thread.start()
        for thread in threads: thread.join()

    def get_response_json_object(self, city, addresses):
        """
        Output: JSON

        Returns response from requested json information
        """
        start, stop = self.geolocation(city)
        avail_data = urllib.urlencode({"lat": start[0], "lng": start[1]})
        cost_data = urllib.urlencode({"start_lat": start[0], "start_lng": start[1],"end_lat": stop[0], "end_lng": stop[1]})

        datatypes = ['ridetypes','eta','drivers','cost']
        all_data = {}
        for dtype in datatypes:
            if dtype != 'cost':
                data = avail_data
            else:
                data = cost_data
            req = urllib2.Request(url=self.url + dtype + '?' + data, headers={"Authorization": "Bearer {}".format(self.access_token)})
            r = urllib2.urlopen(req)
            if r.code != 200:
                print "error requesting document for {}, error: {}".format(dtype, r.read())
            else:
                all_data.update(json.loads(r.read()))

        data = {
            'city': city,
            'start_address': addresses[0],
            'start_latitude': start[0],
            'start_longitude': start[1],
            'stop_address': addresses[1],
            'stop_latitude': stop[0],
            'stop_longitude': stop[1],
            'record_time': time.time(),
            'ridetypes': all_data['ride_types'],
            'eta': all_data['eta_estimates'],
            'drivers': all_data['nearby_drivers'],
            'cost': all_data['cost_estimates']
        }
        collection.insert_one(data)

    def cities_addresses(self):
        """
        Output: Stores all cities addresses into a dictionary
        """
        cities = ['denver','ny','chicago','seattle','sf'] #,'austin','boston']
        '''
        Downtown -> Airport
        '''
        denver = ('301 14th St, Denver, CO 80202','8500 Pena Blvd, Denver, CO 80249')
        ny = ('712 7th Ave, New York, NY 10036','New York, NY 11430')
        chicago = ('77 W Jackson Blvd, Chicago, IL 60604',"A10, O'Hare Commercial Arrivals, Chicago, 60666")
        seattle = ('248 Madison St, Seattle, WA 98104','9075 Airport Expy, SeaTac, WA 98188')
        sf = ('39 Fell St, San Francisco, CA 94102','SFO International Terminal Main Hall, San Mateo County, CA, 94128')
        austin = ('300 W 10th St, Austin, TX 78701','3600 Presidential Boulevard, Austin, TX 78719')
        boston = ('53 Hawley St, Boston, MA 02110','1 Harborside Dr, Boston, MA 02128')

        addresses = [denver] + [ny] + [chicago] + [seattle] + [sf] + [austin] + [boston]

        # store all of the addresses and cities in dictionary
        d = defaultdict(list)
        for i,c in enumerate(cities):
            d[c] = addresses[i]
        return d

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
    lar = LyftAPIRequest()
    lar.run()
