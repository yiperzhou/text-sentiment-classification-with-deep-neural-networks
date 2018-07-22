
import pandas as pd
import json
import os
from overview_data_helper import label_sentiment_category


# def getCollection(colletName = ""):
#     '''
#     return pandas dataframe.
#     '''
#     cursor = db[colletName].find({})
#     df = pd.DataFrame(list(cursor))
#     return df
#
#
# # connect to the db
# client = MongoClient()
# db = client.hotelreviews
#
# # take Barcelona city hotel reviews as example
# cityName = "barcelonaTripadvisor"
# df = getCollection(colletName = cityName)
#
#
# # testCollNames = testCollTripadvisorNames + testCollBookingNames
#
# newDatabaseName = "sentimentAnalysis"
# newDB = client[newDatabaseName]
# # sentAnalysisList = list()
# df.dropna(axis=0, how="any", inplace=True)
# df.drop_duplicates(inplace=True)
# # df.dropna(axis=0, how="any", inplace=True)
# newDB.create_collection("barcelonaTripadvisor")
# odo(df, newDB["barcelonaTripadvisor"])
#
#
#
#
# print("finished")

def get_files_name(folder_path):
    arr = os.listdir(folder_path)
    return arr


def remove_null_sentiment_review(data):
    filtered_df = data[data["review"].notnull()]
    filtered_df = filtered_df[filtered_df["score"].notnull()]
    return filtered_df


def load_raw_tripadvisor_city_data(fileList, folder):
    all_data = []
    for file in fileList:
        p = pd.read_json(folder+os.sep+file, lines=True)
        # with open(folder+os.sep+file,'r') as f:
        #     rawData = json.loads(f)
        all_data.append(p)
    return all_data


def keep_review_score(data):
    newData = pd.DataFrame()
    newData["review"] = data["review"]
    newData["score"] = data["score"]

    return newData


def train_test_split(df, test_percent=0.2):

    train_size = int(df.shape[0] * (1 - test_percent))

    train_df, test_df = df.iloc[:train_size, :], df.iloc[train_size:, :]
    return train_df, test_df



if __name__ == "__main__":
    folder = "/home/yi/sentimentAnalysis/data/rawData/tripadvisor"

    files = get_files_name(folder)
    data_dfs = load_raw_tripadvisor_city_data(files, folder)

    clean_dfs = []
    for data_df in data_dfs:
        clean_dfs.append(remove_null_sentiment_review(data_df))

    new_folder = "/home/yi/sentimentAnalysis/data/no_nan_review_score/tripadvisor/"
    os.makedirs(new_folder, exist_ok=True)

    for fname, f in zip(files, clean_dfs):
        f.to_csv(new_folder+fname)


    # merge all dataframes into one
    df0 = clean_dfs[0]
    for df in clean_dfs[1:]:
        df0 = pd.concat([df0, df], axis=0)

    train_df, test_df = train_test_split(df0)

    train_test_split_folder = "/home/yi/sentimentAnalysis/data/train_test_split/tripadvisor/"
    os.makedirs(train_test_split_folder, exist_ok=True)

    train_df.to_csv(train_test_split_folder + "train_tripadvisor.csv")
    test_df.to_csv(train_test_split_folder + "test_tripadvisor.csv")

    train_test = [train_df, test_df]
    rev_sent_train_test = []
    for df in train_test:
        rev_sent_df = keep_review_score(df)
        sent_list = rev_sent_df.apply(lambda row: label_sentiment_category(row), axis=1)
        rev_sent_df["sentiment"] = sent_list
        rev_sent_df.drop(["score"], axis=1, inplace=True)
        rev_sent_train_test.append(rev_sent_df)

    rev_sent_train_test_folder = "/home/yi/sentimentAnalysis/data/rev_sent_train_test/tripadvisor/"
    os.makedirs(rev_sent_train_test_folder, exist_ok=True)

    rev_sent_train_test[0].to_csv(rev_sent_train_test_folder + "rev_sent_train_tripadvisor.csv")
    rev_sent_train_test[1].to_csv(rev_sent_train_test_folder + "rev_sent_test_tripadvisor.csv")


