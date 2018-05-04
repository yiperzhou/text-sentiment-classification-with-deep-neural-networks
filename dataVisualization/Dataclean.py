import os

from pyspark.sql import SparkSession
import datetime

"""
the aim of this python code is to use py-spark to analyse hotel reviews data and get the basic statistics about the json file
"""


# source file needed to be analysed
dataFile = "Amsterdam_tripadvisor.json"

spark = SparkSession.builder.appName("Python Spark SQL basic example").\
    config("spark.some.config.option", "Amsterdam_tripadvisor").getOrCreate()

df = spark.read.json(dataFile)




df.count()
print(df.filter("date is not null").count())

# filter rows which its date column is null
df = df.filter("date is not null")
df.count()
E = df.select("hotelUrl").distinct().count()

# convert spark dataframe into the resilient distributed dataset
data = df.rdd

ts = datetime.datetime.strptime(data.first()["date"], "%B %d, %Y").date()
te = datetime.datetime.strptime(data.first()["date"], "%B %d, %Y").date()
allDate = data.map(lambda x: x["date"]).collect()

for date in allDate:
    if date == None:
        pass
    else:
        ts = min(ts, datetime.datetime.strptime(date, "%B %d, %Y").date())
        te = max(te, datetime.datetime.strptime(date, "%B %d, %Y").date())
print(ts)
print(te)

# delete records if it isn't complete-format, like absencing values for key

# calculate the popularity of a specific hotel
# t = te -ts
reviewsCount = E

# hotelsList = data.map(lambda x: x["hotelUrl"]).countByValue()
hotelsList = data.map(lambda x: (x["hotelName"], x["hotelUrl"])).countByValue()
# print(hotelsList)


popularity = dict()
i = 0
for hotelInfo, count in hotelsList.items():
    # find the number of reviews for this hotel
    if hotelInfo not in popularity.keys():
        score = float(count)/reviewsCount
        popularity[hotelInfo] = score
    else:
        popularity[hotelInfo] += float(count)/reviewsCount

# sort hotels' popularity in order
popularity = sorted(popularity.items(), key= lambda x: -x[1])


# calculate popularity through the second method, the different users who gave reviews
popularity2 = dict()
hotelsList2 = data.map(lambda x: (x["hotelName"], x["hotelUrl"], x["userId"])).countByValue()

#change the count value into 1, since each user can only be considered given one reviews to one hotel.
#calculate the total unique reviews
reviewsCountBydiffUsers = len(hotelsList2)

for reviewInfo, count in hotelsList2.items():
    if (reviewInfo[0], reviewInfo[1]) not in popularity2.keys():
        popularity2[(reviewInfo[0], reviewInfo[1])] = float(1)/reviewsCountBydiffUsers
    else:
        popularity2[(reviewInfo[0], reviewInfo[1])] += float(1)/reviewsCountBydiffUsers
#sort hotels' popularity in order

popularity2 = sorted(popularity2.items(), key= lambda x: -x[1])

print popularity2







# calculate Sentimentality based on sentstrength
sentiments = dict()
for review in data:






# calculate controversiality










