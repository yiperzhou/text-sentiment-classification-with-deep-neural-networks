# read data from mongodb
from pymongo import MongoClient
import pandas as pd
import numpy as np
import json
from odo import odo
import time
import datetime

def getCollection(colletName = ""):
    '''
    return pandas dataframe.
    '''
    cursor = db[colletName].find({})
    df = pd.DataFrame(list(cursor))
    return df


def convertDate(stringDate):
    '''
    convert string date format into python comparable striptime format
    '''
    return dt.strptime(stringDate, "%B %d, %Y")

# connect to the db
client = MongoClient()
db = client.hotelreviews

# take Barcelona city hotel reviews as example
cityName = "barcelonaTripadvisor"
df = getCollection(colletName = cityName)


# testCollNames = testCollTripadvisorNames + testCollBookingNames

newDatabaseName = "sentimentAnalysis"
newDB = client[newDatabaseName]
# sentAnalysisList = list()
df.dropna(axis=0, how="any", inplace=True)
df.drop_duplicates(inplace=True)
# df.dropna(axis=0, how="any", inplace=True)
newDB.create_collection("barcelonaTripadvisor")
odo(df, newDB["barcelonaTripadvisor"])
# save clean json data into file
# df.to_json("barcelonaTripadvisor.json", orient = "records")
# except:
#     print("error")
#     pass



print("finished")