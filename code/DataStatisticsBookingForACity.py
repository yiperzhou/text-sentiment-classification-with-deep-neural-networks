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
    spark = SparkSession.builder.appName("statistics for hotels reviews in Booking").config(
    "spark.some.config.option", "booking").getOrCreate()
    inputfile, outputfile = getArgsFromCommand()
    data = spark.read.json(inputfile)
    data = data.drop_duplicates()   # try to remove duplicate reviews

    statistics = dict()
    statistics["totalReviews"] = data.select(data.url).distinct().count()
    statistics["totalHotels"] = data.select(data.hotelUrl).distinct().count()
    statistics["avgReviewNumAHotel"] =  statistics["totalReviews"] / statistics["totalHotels"]
    statistics["0 stars"] = 0
    statistics["1 stars"] = 0
    statistics["2 stars"] = 0
    statistics["3 stars"] = 0
    statistics["4 stars"] = 0
    statistics["5 stars"] = 0
    statistics["null stars"] = 0
    statistics["eachHotelDetail"]  = list()
    statistics["minReviewNumAHotel"] = statistics["totalReviews"]
    statistics["maxReviewNumAHotel"] = 0

    nullScoreNumAHotel = 0
    nullScoreNum = 0
    totalScoreAHotel = 0
    totalScore = 0
    # get statistics for each single hotel
    for row in data.groupby(data.hotelUrl).count().collect():
        oneHotel = dict()
        oneHotel["reviewsNum"] = row["count"]
    #     print(oneHotel["reviewsNum"])
        if oneHotel["reviewsNum"] > statistics["maxReviewNumAHotel"]:
            statistics["maxReviewNumAHotel"] = oneHotel["reviewsNum"]
        if oneHotel["reviewsNum"] < statistics["minReviewNumAHotel"]:
            statistics["minReviewNumAHotel"] = oneHotel["reviewsNum"]
        oneHotel["hotelUrl"] = row["hotelUrl"]
        try:
            oneHotel["hotelStars"] = data.filter(data.hotelUrl == row["hotelUrl"]).first()["hotelStars"]
        except:
            oneHotel["hotelStars"] = None
        oneHotel["0 stars"] = 0
        oneHotel["1 stars"] = 0
        oneHotel["2 stars"] = 0
        oneHotel["3 stars"] = 0
        oneHotel["4 stars"] = 0
        oneHotel["5 stars"] = 0
        
        for record in data.filter(data.hotelUrl == row["hotelUrl"]).collect():
            if record["score"] == None:
                nullScoreNumAHotel += 1
                nullScoreNum += 1
            else:
    #             remove \n newline and convert string into float
                try:
                    totalScore += float(record["score"])
                    totalScoreAHotel += float(record["score"])
                except:
                    nullScoreNumAHotel += 1
                    nullScoreNum += 1
                    pass
    #             apply Department of Tourism (DOT) Star Grading System;
    # 5 stars - (0.85, 10]; 4 stars - (0.7, 0.85]; 3 stars - (0.55, 0.7]; 2 stars - (0.4, 0.55]; 1 stars - (0.25, 0.4]
    # method: score/10*5; then round.
                oneHotel[str(round(float(record["score"])/10 * 5)) + " stars"] += 1
                statistics[str(round(float(record["score"])/10 * 5)) + " stars"] += 1
        if (oneHotel["reviewsNum"] - nullScoreNumAHotel) == 0:
            oneHotel["avgScore"] = None #means all reviews aren't gaven scores.
        else:
            oneHotel["avgScore"] = totalScoreAHotel / (oneHotel["reviewsNum"] - nullScoreNumAHotel)
        nullScoreNumAHotel = 0
        statistics["eachHotelDetail"].append(oneHotel)
    statistics["null stars"] = nullScoreNum
    statistics["avgScore"] = totalScore / (statistics["totalReviews"] - nullScoreNum)
    with open(outputfile, "w") as f:
        json.dump(statistics, f)
        f.close()