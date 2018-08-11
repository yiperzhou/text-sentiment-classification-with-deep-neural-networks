import os
import sys
import json


def DataStatisticsBookingForACity():

    spark = SparkSession.builder.appName("statistics for hotels reviews in Booking").config(
        "spark.some.config.option", "booking").getOrCreate()
    inputfile, outputfile = getArgsFromCommand()
    data = spark.read.json(inputfile)
    data = data.drop_duplicates()  # try to remove duplicate reviews

    statistics = dict()
    statistics["totalReviews"] = data.select(data.url).distinct().count()
    statistics["totalHotels"] = data.select(data.hotelUrl).distinct().count()
    statistics["avgReviewNumAHotel"] = statistics["totalReviews"] / statistics["totalHotels"]
    statistics["0 stars"] = 0
    statistics["1 stars"] = 0
    statistics["2 stars"] = 0
    statistics["3 stars"] = 0
    statistics["4 stars"] = 0
    statistics["5 stars"] = 0
    statistics["null stars"] = 0
    statistics["eachHotelDetail"] = list()
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

                oneHotel[str(round(float(record["score"]) / 10 * 5)) + " stars"] += 1
                statistics[str(round(float(record["score"]) / 10 * 5)) + " stars"] += 1
        if (oneHotel["reviewsNum"] - nullScoreNumAHotel) == 0:
            oneHotel["avgScore"] = None  # means all reviews aren't gaven scores.
        else:
            oneHotel["avgScore"] = totalScoreAHotel / (oneHotel["reviewsNum"] - nullScoreNumAHotel)
        nullScoreNumAHotel = 0
        statistics["eachHotelDetail"].append(oneHotel)
    statistics["null stars"] = nullScoreNum
    statistics["avgScore"] = totalScore / (statistics["totalReviews"] - nullScoreNum)
    with open(outputfile, "w") as f:
        json.dump(statistics, f)
        f.close()


def DataStatisticsTripadvisorForACity():
    spark = SparkSession.builder.appName("hotel reviews statistics").config("spark.some.config.option",
                                                                            "tripadvisor").getOrCreate()
    inputfile, outputfile = getArgsFromCommand()
    data = spark.read.json(inputfile)
    data = data.drop_duplicates()  # try to remove duplicate reviews

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

    nullScoreReviews = 0  # store the number which don't give score for all hotels reviews in the city

    # calculate statistics for each hotel in this city
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
        # add total score that each hotel gets
        totalScore += totalScoreForOneHotel
        try:
            hotelDetails["avgScore"] = totalScoreForOneHotel / (hotelDetails["reviewsNum"] - nullScoreReviewsAHotel)
        except:
            pass
        totalScoreForOneHotel = 0
        statistics["eachHotelDetail"].append(hotelDetails)

    statistics["avgScore"] = totalScore / (statistics["totalReviews"] - nullScoreReviews)

    # classify all reviews regardless hotels into 6 categories, 0, 1, 2, 3, 4, 5 scores
    for scoreCityRow in data.groupBy(data.score).count().collect():
        if scoreCityRow["score"] == None:
            kCity = "null score"
        else:
            kCity = str(scoreCityRow["score"]) + " score"
        statistics[kCity] = scoreCityRow["count"]
    with open(outputfile, "w") as f:
        json.dump(statistics, f)
        f.close()

