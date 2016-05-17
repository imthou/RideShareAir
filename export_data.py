from pymongo import MongoClient
import os
import pandas as pd

ec2host = os.environ['EC2HOST']
client = MongoClient(host=ec2host,
                        port=27017)
uberdb = client['apidata']
ubercoll = uberdb['uberapi']
lyftdb = client['lyftdata']
lyftcoll = lyftdb['lyftapi']

if __name__ == '__main__':
    start_date = (pd.to_datetime('2016-05-09') + pd.Timedelta(hours=7)).value // 10**9
    end_date = (pd.to_datetime('2016-05-16') + pd.Timedelta(hours=7)).value // 10**9
    docs = ubercoll.find({'record_time':{'$gte':start_date,
                                    '$lte':end_date}},
                            {'record_time': 1, 'city':1, 'prices':1, '_id':0})
