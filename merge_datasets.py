import pandas as pd
import sys

def main():
    """
    Output: CSV file

    Merges most recent Uber/Lyft Data
    """
    maindf = pd.read_csv(sys.argv[1], parse_dates=['record_time'])
    newdf = pd.read_csv(sys.argv[2], parse_dates=['record_time'])
    mergedf = pd.concat([maindf,newdf])
    mergedf.to_csv(sys.argv[3], index=False)
    return mergedf

if __name__ == '__main__':
    mergedf = main()
    print mergedf['record_time'].head()
    print mergedf['record_time'].tail()
