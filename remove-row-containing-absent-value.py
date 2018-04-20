
# coding: utf-8

# In[2]:


# read data from mongodb
from pymongo import MongoClient
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession


# In[3]:


# connect to mongodb in localhost and access database and collections
def _connect_mongo():
    """ A util for making a connection to mongo """
    host = "localhost"
    port = 27017
    username = ""
    password = ""
    db = "5gopt"

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]


# In[4]:


# get specific colelction, and remove the "_id" collumn
def collection_read_mongo(collection, query={}, no_id = True):
    db = _connect_mongo()
    cursor = db[collection].find(query)
    df = pd.DataFrame(list(cursor))

    if no_id:
        try:
            del df["_id"]
        except:
            pass
    return df


# In[5]:


citiesName = ["Amsterdam", "Athens", "Barcelona", "Berlin", "Helsinki", "Paris", "Stockholm", 
              "London", "Manchester", "Roma", "Prague", "Edinburgh", "Vienna", "Lisbon", "Budapest",
             "Madrid", "Warsaw", "Brussels", "Dublin", "Munich", "Riga", "hamburg",
             "Vilnius", "Tallinn", "Reykjavik", "Frankfurt", "Zurich", 
              "liverpool", "Minsk", "Oslo", "Kiev", "Bucharest", "Sofia", "Cologne", "Kharkiv"]


# In[6]:


len(citiesName)


# In[7]:


"Roma" in citiesName


# In[8]:


# convert cities name into all-lowercase name, and then check whether there are any duplicating names
citiesName = [i.lower() for i in citiesName]
duplicate = [i for i in citiesName if citiesName.count(i) > 1]
if len(duplicate) > 1:
    print("the list of cities name doesn't contain all cities, CHECK, CHECK")

citiesName = set(citiesName)


# In[9]:


citiesName


# In[10]:


# iterately access each mongodb collection and compare their data size, then rank them in order
client = MongoClient()


# In[11]:


db = client.test


# In[12]:


def changeCollectionName():
    '''
    this function only works for changing mongo collection name
    '''
    collNames = db.collection_names()
    for i in collNames:
        subNameList = i.split("_")
        if len(subNameList) > 1:
            if subNameList[1] in ["tripadvisor", "Tripadvisor"]:
#                 print("True")
                newName = subNameList[0].lower() + "Tripadvisor"
            if subNameList[1] in ["Booking", "booking"]:
                newName = subNameList[0].lower() + "Booking"
            else:#                 using original nameri
                newName = "_".join(subNameList)
            db[i].rename(newName)
        else:
            pass
    return db.collection_names()


# In[13]:


print(db.collection_names())
tripCollNames = list()
bookingCollNames = list()


# In[14]:


tripadvisorSuffix = "Tripadvisor"
bookingSuffix = "Booking"
dataSizeList = list()
ByteToMB = 1024*1024
for i in citiesName:
    if i == "":
        pass
    else:
#         print(citiesName)
#         print(i)
        collTripadvisor = i+tripadvisorSuffix
        tripCollNames.append(collTripadvisor)
        collBooking = i+bookingSuffix
        bookingCollNames.append(collBooking)
    #     get the collection size and then convet to MB from bytes
    #     print(collTripadvisor)
#     since for booking or tripadvisor, some cities data doesn't be collected.
        try:
            size_in_trip = db.command("collstats", collTripadvisor)["size"] / ByteToMB
            countNum_in_trip = db.command("collstats", collTripadvisor)["count"]
        except:
            size_in_trip = None
            countNum_in_trip = None
#             pass
        try:
            size_in_booking = db.command("collstats", collBooking)["size"] / ByteToMB
            countNum_in_booking = db.command("collstats", collBooking)["count"]
        except:
            size_in_booking = None
            countNum_in_booking = None
#             pass
        dataSizeList.append([i, size_in_trip, countNum_in_trip, size_in_booking, countNum_in_booking])

dataSizeDF = pd.DataFrame(dataSizeList, columns=["CityName", "Tripadvisor(MB)","CountInTrip", "Booking(MB)", "CountInBooking"])
    
    


# In[15]:


dataSizeDF


# In[16]:


# sort dataframe only according to column "Booking.com"
top10Trpadvisor = dataSizeDF.sort_values(by="Tripadvisor(MB)", ascending=False)[:10][["CityName", "Tripadvisor(MB)", "CountInTrip"]]
top10Trpadvisor


# In[17]:


# sort dataframe only according to column "Booking.com"
top10Booking = dataSizeDF.sort_values(by="Booking(MB)", ascending=False)[:10][["CityName", "Booking(MB)", "CountInBooking"]]
top10Booking


# In[ ]:





# In[ ]:


#do statistics about review count. 
testCollNames = ["amsterdamTripadvisor"]
newDatabaseName = "sentimentAnalysis"
sentAnalysisList = list()
for i in testCollNames:
#     print(type(db[i]))
#     fetch collection data from mongo, and then calcualte the statistics, we can do it just within mongo, which
# will be more efficient, but we also need these data stored in new dataset, sentimentAnalysis. So we should 
# get collections first.
    cursor = db[i].find({})
    df = pd.DataFrame(list(cursor))
#     dataDF = pd.DataFrame(db[i].find({}))
    print("the original dataframe shape:    ", df.shape)
    df.drop_duplicates(inplace=True)
    print("the dataframe shape after removing duplication:    ", df.shape)
    df.dropna(axis=0, how="any", inplace=True)
    print("the dataframe shape after removing any null column values:    ", df.shape)
    
    sentAnalysisList.append([i ,df.shape[0]])
    newDB = client[newDatabaseName]
    newDB[i].insert_many(df.to_json())
    
#     print(df[:1])
#     print(df[:1]["date"])
#     print(type(df))


sentAnalysisDF = pd.DataFrame(sentAnalysisList, columns=["CollName", "TripadvisorCount"])

    
    
    

    
    

# remove review records which contain absent value

# print sample json data in booking json raw data

# sample json data in tripadvisor


# In[ ]:





# In[ ]:





# In[32]:


print(db.command("collstats", "amsterdamTripadvisor")["size"])


# In[10]:





# In[ ]:


spark = SparkSession.builder.appName("statistics for hotels reviews in Booking").config(
    "spark.some.config.option", "booking").getOrCreate()

