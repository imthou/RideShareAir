import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

class LyftParse(object):
    """
    Parses Lyft data and organizes it into a Dataframe
    """
    def __init__(self):
        """
        Output: List of dictionaries

        Loads JSON documents and transforms to list of dictionaries
        """
        filename = "data/lyft1_41416.json"
        with open(filename) as f:
            self.data = [json.loads(line) for line in f]

    def parse_lyft_data(self):
        """
        Output: Dataframe

        Parses each document and organizes it into a dataframe
        """
        dfs = []
        for i, d in enumerate(self.data):
            df = pd.DataFrame(d['cost'])
            try:
                df = df.merge(pd.DataFrame(d['drivers']), left_on="ride_type", right_on="ride_type")
            except:
                pass
            df = df.merge(pd.DataFrame(d['eta']), left_on=["display_name","ride_type"], right_on=["display_name","ride_type"])
            rt = pd.DataFrame(d['ridetypes'])
            rt = pd.concat([rt,pd.DataFrame(rt['pricing_details'].tolist())],axis=1).drop('pricing_details', axis=1)
            df = df.merge(rt, left_on=["display_name","ride_type","currency"], right_on=["display_name","ride_type","currency"], how="outer")
            try:
                df['num_drivers'] = df['drivers'].apply(lambda x: len(x))
            except:
                df['num_drivers'] = 0
            for info in ['city','record_time','start_latitude','start_longitude','stop_latitude','stop_longitude','start_address','stop_address']:
                df[info] = d[info]
            df['record_time'] = pd.to_datetime(df['record_time'], unit='s')
            df.drop('display_name', axis=1, inplace=True)
            dfs.append(df)
            if i % 50 == 0:
                print "progress: {}, {:.2f}%".format(i, float(i)/len(data) * 100.0)
        self.df = pd.concat(dfs)

    def save_data_csv(self):
        """
        Output: CSV file

        Saves Dataframe to CSV file
        """
        filename = "data/lyft_data.csv"
        self.df.to_csv(filename)
        print "saved file to {}".format(filename)

if __name__ == '__main__':
    lp = LyftParse()
    lp.parse_lyft_data()
    lp.save_data_csv()
