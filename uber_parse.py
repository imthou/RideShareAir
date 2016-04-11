import pandas as pd
import json
from pprint import pprint
from pandas.io.json import json_normalize
import sys

class UberParse(object):
    """
    Class for converting Uber json documents to a pandas Dataframe
    """

    def __init__(self, filename):
        """
        Input: Store the filename
        """
        self.filename = filename

    def run(self):
        """
        Output: Runs the parsing and merging functions and exports the dataframe into a csv file
        """
        data = self.parse_json()
        self.merge_dfs(data)
        self.df = pd.concat(self.dfs)
        self.df.drop(['price_details'],axis=1,inplace=True)
        self.df.to_csv('data/uber_data_{}.csv'.format(self.filename.split('_')[1].split('.')[0]), index=False, encoding='utf-8')

    def parse_json(self):
        """
        Output: List of Json Documents

        Loads all of the Mongodb documents and converts them into json documents
        """
        with open(self.filename) as f:
            data = [json.loads(doc) for doc in f]
        return data

    def merge_dfs(self, data):
        """
        Input: List of Json documents
        Output: DataFrame

        Merged dataframe of Uber data
        """
        keys = ['products:','times','prices']
        dfs = []
        for i,d in enumerate(data):
            record = pd.DataFrame({k:v for k,v in d.iteritems() if k not in keys}, index=[0]).drop('_id',axis=1)
            try:
                products, times, prices = [json_normalize(d[k]) for k in keys]
                times.drop(['display_name','localized_display_name'],axis=1,inplace=True)
                prices.drop(['currency_code','display_name','localized_display_name'],axis=1,inplace=True)
                others = products.merge(times, left_on='product_id', right_on='product_id').merge(prices, left_on='product_id', right_on='product_id')
                others['city'] = record['city']
                others['city'] = others['city'].apply(lambda x: others['city'][0])
                dfs.append(others.merge(record, how='outer'))
            except:
                print "error in json_normalize"
            if i % 50 == 0:
                print "progress: {}, {:.2f}%".format(i, float(i)/len(data) * 100.0)
        self.dfs = dfs

if __name__ == '__main__':
    filename = sys.argv[1]
    up = UberParse(filename=filename)
    up.run()
