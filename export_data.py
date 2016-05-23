from pymongo import MongoClient
import os
import pandas as pd
import sys

# ec2host = os.environ['EC2HOST']
# client = MongoClient(host=ec2host,
#                         port=27017)
# uberdb = client['apidata']
# ubercoll = uberdb['uberapi']
# lyftdb = client['lyftdata']
# lyftcoll = lyftdb['lyftapi']

if __name__ == '__main__':
    # Monday to Monday, e.g. 2016-05-16 to 2016-05-23
    start_date = (pd.to_datetime(sys.argv[1]) + pd.Timedelta(hours=7)).value // 10**9
    end_date = (pd.to_datetime(sys.argv[2]) + pd.Timedelta(hours=7)).value // 10**9
    filename = sys.argv[3]
    datatype = sys.argv[4]
    print "start date value: {}".format(start_date)
    print "end date value: {}".format(end_date)
    if datatype == "uber":
        command = """mongoexport --db apidata --collection uberapi --query '{"record_time" : {"$gte": %d, "$lte": %d}}' --out %s""" % (start_date,end_date,filename)
    else:
        command = """mongoexport --db lyftdata --collection lyftapi --query '{"record_time" : {"$gte": %d, "$lte": %d}}' --out %s""" % (start_date,end_date,filename)
    os.system(command)

    """
    docs = ubercoll.find({'record_time':{'$gte':start_date,
                                    '$lte':end_date}},
                            {'record_time': 1, 'city':1, 'prices':1, '_id':0})
    db.uberapi.find({"record_time" : {"$gte": new Date(2016, 4, 16).getTime() / 1000 + 25200,"$lte": new Date(2016, 4, 23).getTime() / 1000 + 25200}}).count()

    mongoexport --db apidata --collection uberapi --query '{"record_time" : {"$gte": new Date(2016, 4, 9).getTime() / 1000 + 25200, "$lte": new Date(2016, 4, 16).getTime() / 1000 + 25200 }}' --out uber9_50916.json

    mongoexport --db apidata --collection uberapi --query '{"record_time" : {"$gte": 1462777200, "$lte": 1463382000}}' --out uber9_50916.json
    """
