from pymongo import MongoClient
import pandas as pd
import os

ec2host = os.environ['EC2HOST']

client = MongoClient(host=ec2host,
                        port=27017)
db = client['apidata']
collection = db['uberapi']

if __name__ == '__main__':
    start_date = (pd.to_datetime('2016-04-11') + pd.Timedelta(hours=7)).value // 10**9
    end_date = (pd.to_datetime('2016-04-17') + pd.Timedelta(hours=7)).value // 10**9
    docs = collection.find({'record_time':{'$gte':start_date,
                                    '$lte':end_date}},
                            {'record_time': 1, 'city':1, 'prices':1, '_id':0})
    data = []
    for doc in docs:
        df = pd.DataFrame(doc['prices'])
        df['avg_price_est'] = (df['low_estimate'] + df['high_estimate']) / 2.
        df['record_time'] = pd.to_datetime(doc['record_time'], unit='s')
        df['city'] = doc['city']
        df['display_name'].replace(['UberBLACK','UberSUV','UberSELECT','uberT','Yellow WAV','ASSIST','PEDAL','For Hire','#UberTAHOE','uberCAB','WarmUpChi'],
                           ['uberBLACK','uberSUV','uberSELECT','uberTAXI','uberWAV','uberASSIST','uberPEDAL','uberTAXI','uberTAHOE','uberTAXI','uberWARMUP'], inplace=True)
        df['display_name'] = df['display_name'].apply(lambda x: x.lower())
        data.append(df[['record_time','city','display_name','avg_price_est']])
    df = pd.concat(data)
    df = df.set_index('record_time')
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    hourly = df.groupby(['date','hour','city','display_name']).mean().reset_index()
    hourly['record_time'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ":00:00")
    hourly.set_index('record_time', inplace=True)
    # print collection.find_one()
