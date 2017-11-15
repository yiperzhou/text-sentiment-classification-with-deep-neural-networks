"""
this file calculate the basic reviews statistics for each different cities
"""

"""
env: python3.5, py-spark-version 2.1.1, jdk-1.8
in terminal run below command to execute the file:
python DataStatistics.py -i [inputfilename.json] -o [outputfilename.txt]
"""
import os
import sys
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql import Row

import argparse
import json


def getArgsFromCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="inputfilename.json")
    parser.add_argument("-o", help="outputfilename.txt")
    args = parser.parse_args()
    try:
        inputfile = args.i
        outputfile = args.o
        return inputfile, outputfile
    except args as identifier:
        print("input parameter error, input like: python DataStatistics.py -i [inputfilename.json] -o [outputfilename.txt]")
        return


if __name__ == "__main__":
    spark = SparkSession.builder.appName("hotel reviews statistics").config("spark.some.config.option", "tripadvisor").getOrCreate()
    inputfile, outputfile = getArgsFromCommand()
    data = spark.read.json(inputfile)
    data = data.drop_duplicates()   # try to remove duplicate reviews

    statistics = dict()

    statistics["totalReviews"] = data.select("url").distinct().count()
    statistics["maxReviewNumAHotel"] = 0
    statistics["minReviewNumAHotel"] = statistics["totalReviews"]

    statistics["eachHotelDetail"] = list()

    hotelsList = data.groupBy(data.hotelUrl).count().collect()
    statistics["totalHotels"] = len(hotelsList)
    statistics["avgReviewNumAHotel"] = statistics["totalReviews"] / len(hotelsList)

    totalScore = 0
    totalScoreForOneHotel = 0
    
    nullScoreReviews = 0    #store the number which don't give score for all hotels reviews in the city

    #calculate statistics for each hotel in this city
    for row in hotelsList:
        hotelDetails = dict()
        hotelDetails["hotelUrl"] = row["hotelUrl"]

        oneHotelData = data.filter(data.hotelUrl == row["hotelUrl"])
        
        try:
    #         hotelStars means hotel class for this hotel, not the review score.
            hotelDetails["hotelStars"] = data.filter(data.hotelUrl == row["hotelUrl"]).first()["hotelStars"]
        except:
            hotelDetails["hotelStars"] = None
        
        try:
            hotelDetails["reviewsNum"] = oneHotelData.count()
            if hotelDetails["reviewsNum"] > statistics["maxReviewNumAHotel"]:
                statistics["maxReviewNumAHotel"] = hotelDetails["reviewsNum"]
            if hotelDetails["reviewsNum"] < statistics["minReviewNumAHotel"]:
                statistics["minReviewNumAHotel"] = hotelDetails["reviewsNum"]
        except:
            pass
        
        nullScoreReviewsAHotel = 0
        hotelDetails["avgScore"] = 0
        try:
            for scoreRow in oneHotelData.groupBy(oneHotelData.score).count().collect():
        #         if the user doesn't give any score, then 
                if scoreRow["score"] == None:
                    k = "null score"
                    nullScoreReviews += 1
                    nullScoreReviewsAHotel += 1
                else:
                    k = str(scoreRow["score"]) + " score"
                    totalScoreForOneHotel += float(scoreRow["score"]) * float(scoreRow["count"])
                hotelDetails[k] = scoreRow["count"]
        except:
            pass
        #add total score that each hotel gets
        totalScore += totalScoreForOneHotel
        try:
            hotelDetails["avgScore"] = totalScoreForOneHotel / (hotelDetails["reviewsNum"] - nullScoreReviewsAHotel)    
        except:
            pass
        totalScoreForOneHotel = 0
        statistics["eachHotelDetail"].append(hotelDetails)

    statistics["avgScore"] = totalScore / (statistics["totalReviews"] - nullScoreReviews)

#classify all reviews regardless hotels into 6 categories, 0, 1, 2, 3, 4, 5 scores
    for scoreCityRow in data.groupBy(data.score).count().collect():
        if scoreCityRow["score"] == None:
            kCity = "null score"
        else:
            kCity = str(scoreCityRow["score"]) + " score"
        statistics[kCity] = scoreCityRow["count"]
    with open(outputfile, "w") as f:
        json.dump(statistics, f)
        f.close()



