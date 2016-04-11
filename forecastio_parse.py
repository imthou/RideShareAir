import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

class ForecastParse(object):
    """
    Cleaning forecast data and throwing it into a dataframe
    """

    def __init__(self):
        """
        Read in weather json file into dataframe
        """
        with open('data/weather_021516_033016.json') as f:
            data = [json.loads(doc) for doc in f]
        self.df = pd.DataFrame(data).drop(['_id','currently','flags'], axis=1)

    def run(self):
        """
        Output: CSV file

        Exports weather dataframe to csv
        """
        df = self.organized_json(self.df)
        df.to_csv('data/weather_021516_033016_organized.csv', index=False, encoding='utf-8')

    def organized_json(self, df):
        """
        Output: Dataframe

        Merges weather data by hour and city
        """
        df['daily'] = df['daily'].apply(lambda x: json_normalize(x['data'])).apply(lambda x: x.rename(columns=lambda x: 'daily_' + x))
        dai = df['daily'][0]
        for d in df['daily'][1:]:
            dai = dai.append(d)
        dai = dai.reset_index().drop('index', axis=1)
        df = pd.concat([df,dai], axis=1).drop('daily', axis=1)
        df['hourly'] = df['hourly'].apply(lambda x: json_normalize(x['data'])).apply(lambda x: x.rename(columns=lambda x: 'hourly_' + x))
        hou = df['hourly'][0]
        hou['city'] = df['city'][0]
        for i,h in enumerate(df['hourly'][1:]):
            h['city'] = df['city'][i+1]
            hou = hou.append(h)
        hou = hou.reset_index().drop('index', axis=1)
        hou['hourly_time'] = pd.to_datetime(hou['hourly_time'], unit='s')
        hou = hou.set_index('hourly_time')

        # each date is based on their timezone
        # correct each timezone to mountain time
        den = hou.query("city == 'denver'")
        dif = den.index[0].hour
        den.index = den.index - pd.Timedelta(hours=dif)
        for city in ['seattle','sf','ny','chicago']:
            df_c = hou.query("city == @city")
            dif = df_c.index[0].hour
            df_c.index = df_c.index - pd.Timedelta(hours=dif)
            den = den.append(df_c)
        den['date'] = den.index.date

        df.drop('hourly', axis=1, inplace=True)
        df['daily_time'] = pd.to_datetime(df['daily_time'], unit='s')
        df['date'] = df['daily_time'].dt.date
        df2 = den.reset_index().merge(df, left_on=['city','date'], right_on=['city','date'], how='outer')

        return df2

if __name__ == '__main__':
    fp = ForecastParse()
    fp.run()
