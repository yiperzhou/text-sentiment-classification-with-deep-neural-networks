import pandas as pd
import json
import os
from data_analysis_helper import label_sentiment_category



def get_clean_file_names():

    booking_path = "data/raw_revews_data/BookingHotelReviewRawData"
    tripadvisor_path = "data/raw_revews_data/TripadvisorHotelReivewRawData"

    booking_files = [x for x in os.listdir(booking_path)]
    tripadvisor_files = [x for x in os.listdir(tripadvisor_path)]

    csv_booking_save_folder = "data/remove_null_review_score_data/BookingHotelReviewData"
    csv_tripadvisor_save_folder = "data/remove_null_review_score_data/TripadvisorHotelReivewData"

    # Since the format of the review in the booking dataset contains two seperated parts,
    # the negative content and the positive content; the booking dataset is not going to use in this experiments.
    # for raw_file in booking_files:
    #     raw_json_df = pd.read_json(booking_path + os.sep + raw_file, lines=True)
    #     df = remove_null_sentiment_review(raw_json_df)
    #     df.to_csv(csv_booking_save_folder + os.sep + raw_file.split('.')[0]+".csv")

    for raw_file in tripadvisor_files:
        raw_json_df = pd.read_json(tripadvisor_path + os.sep + raw_file, lines=True)
        df = remove_null_sentiment_review(raw_json_df)
        df.to_csv(csv_tripadvisor_save_folder + os.sep + raw_file.split('.')[0]+".csv")



def remove_null_sentiment_review(data):
    filtered_df = data[data["review"].notnull()]
    filtered_df = filtered_df[filtered_df["score"].notnull()]
    return filtered_df


def merge_csv_to_one():
    '''
    merge all files under folder TripadvisorHotelReviewData into one big csv file
    :return:
    '''
    csv_tripadvisor_save_folder = "data/remove_null_review_score_data/TripadvisorHotelReivewData"
    tripadvisor_files = [x for x in os.listdir(csv_tripadvisor_save_folder)]
    all_dfs = []
    for raw_file in tripadvisor_files:
        df = pd.read_csv(csv_tripadvisor_save_folder + os.sep + raw_file)

        all_dfs.append(df)

    final_df = pd.concat(all_dfs, axis=0)
    final_df.to_csv(csv_tripadvisor_save_folder + os.sep + "tripadvisor_review_data.csv")



def split_train_test_dataset():
    data_folder = "data/remove_null_review_score_data/TripadvisorHotelReivewData/"
    df = pd.read_csv(data_folder + os.sep + "tripadvisor_review_data.csv")
    from sklearn.model_selection import train_test_split
    trainingSet, testSet = train_test_split(df, test_size=0.2)
    trainingSet.to_csv(data_folder + os.sep + "tripadvisor_train_dataset.csv")
    testSet.to_csv(data_folder + os.sep + "tripadvisor_test_dataset.csv")



def keep_review_score(data):
    newData = pd.DataFrame()
    newData["review"] = data["review"]
    newData["score"] = data["score"]

    return newData

if __name__ == "__main__":
    # get_clean_file_names()
    # merge_csv_to_one()
    split_train_test_dataset()
    print("done")

