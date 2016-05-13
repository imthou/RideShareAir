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

        Store filename, keys of dictionaries, and empty dataframe
        """
        self.filename = filename
        self.keys = ['products:','times','prices']
        self.dfs = []

    def run(self):
        """
        Output: Runs the parsing and merging functions and exports the dataframe into a csv file
        """
        self._parse_json()
        self._merge_dfs()
        self.df = pd.concat(self.dfs)
        self.df.drop(['price_details'],axis=1,inplace=True)
        self.df.to_csv('data/uber_data_{}.csv'.format(self.filename.split('_')[1].split('.')[0]), index=False, encoding='utf-8')
        print "finished parsing uber data to {}".format(self.filename.split('_')[1].split('.')[0])

    def _parse_json(self):
        """
        Output: List of Json Documents

        Loads all of the Mongodb documents and converts them into json documents
        """
        with open(self.filename) as f:
            self.data = [json.loads(doc) for doc in f]

    def _merge_dfs(self):
        """
        Input: List of Json documents
        Output: DataFrame

        Merged dataframe of Uber data
        """
        for i,d in enumerate(self.data):
            self.record = pd.DataFrame({k:v for k,v in d.iteritems() if k not in self.keys}, index=[0]).drop('_id',axis=1)
            try:
                self.products, self.times, self.prices = [json_normalize(d[k]) for k in self.keys]
                self.times.drop(['display_name','localized_display_name'],axis=1,inplace=True)
                self.prices.drop(['currency_code','display_name','localized_display_name'],axis=1,inplace=True)
                self.others = self.products.merge(self.times, left_on='product_id', right_on='product_id').merge(self.prices, left_on='product_id', right_on='product_id')
                self.others['city'] = self.record['city']
                self.others['city'] = self.others['city'].apply(lambda x: self.others['city'][0])
                self.dfs.append(self.others.merge(self.record, how='outer'))
            except:
                print "error in json_normalize"
            if i % 50 == 0:
                print "progress: {}, {:.2f}%".format(i, float(i)/len(self.data) * 100.0)

if __name__ == '__main__':
    filename = sys.argv[1]
    up = UberParse(filename=filename)
    up.run()
