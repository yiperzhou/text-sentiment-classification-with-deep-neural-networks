
import pandas as pd
import json
import os
from overview_data_helper import label_sentiment_category


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


def change_mongo_collection_name(db=None.test):
    '''
    changing mongo collection name,
    :param db: mongoDB database, default "test" database
    :return:
    '''
    # 35 cities name
    citiesName = ["Amsterdam", "Athens", "Barcelona", "Berlin", "Helsinki",
                  "Paris", "Stockholm", "London", "Manchester", "Roma",
                  "Prague", "Edinburgh", "Vienna", "Lisbon", "Budapest",
                  "Madrid", "Warsaw", "Brussels", "Dublin", "Munich",
                  "Riga", "hamburg","Vilnius", "Tallinn", "Reykjavik",
                  "Frankfurt", "Zurich", "liverpool", "Minsk", "Oslo",
                  "Kiev", "Bucharest", "Sofia", "Cologne", "Kharkiv"]

    collNames = db.collection_names()
    for i in collNames:
        subNameList = i.split("_")
        if len(subNameList) > 1:
            if subNameList[1] in ["tripadvisor", "Tripadvisor"]:
                newName = subNameList[0].lower() + "Tripadvisor"
            if subNameList[1] in ["Booking", "booking"]:
                newName = subNameList[0].lower() + "Booking"
            else:#                 using original nameri
                newName = "_".join(subNameList)
            db[i].rename(newName)
        else:
            pass
    return db.collection_names()


def generate_train_test_data():

    # file1 = "amsterdamTripavisor.csv"
    # file2 = "/home/yi/sentimentAnalysis/data/athensTripavisor.csv"
    # file3 = "/home/yi/sentimentAnalysis/data/barcelonaTripadvisor.csv"
    # file4 = "/home/yi/sentimentAnalysis/data/dublinTripavisor.csv"
    #
    #
    # df1 = pd.read_csv(file1)
    # df2 = pd.read_csv(file2)
    # df3 = pd.read_csv(file3)
    # df4 = pd.read_csv(file3)
    #
    # df = pd.concat([df1, df2, df3, df4], axis=0)
    # df.reset_index(drop=True)
    # assert df.shape[0] == (df1.shape[0] + df2.shape[0] + df3.shape[0] + df4.shape[0])
    # df.to_csv("raw_tripadvisor_data.csv")
    #
    #
    # test_percent = 0.2
    # train_size = int(df.shape[0] * (1 - test_percent))
    #
    # train_df, test_df = df.iloc[:train_size, :], df.iloc[train_size:, :]
    # train_df.to_csv("raw_tripadvisor_data_train.csv")
    # test_df.to_csv("raw_tripadvisor_data_test.csv")
    #
    #
    # # reserve review and sentiment two column
    # clean_train_df = pd.DataFrame()
    # clean_train_df["review"] = train_df["review"]
    # clean_train_df['sentiment'] = train_df.apply(lambda row: label_sentiment_category(row), axis=1)
    #
    # clean_test_df = pd.DataFrame()
    # clean_test_df["review"] = test_df["review"]
    # clean_test_df['sentiment'] = test_df.apply(lambda row: label_sentiment_category(row), axis=1)
    #
    # print("only sentiment and reviews")
    #
    #
    # clean_train_df.to_csv("./csv/train_tripadvisor_5cities.csv")
    # clean_test_df.to_csv("./csv/test_tripadvisor_5cities.csv")
    #
    file_train = "/home/yi/sentimentAnalysis/data/train_test_split/tripadvisor/train_tripadvisor.csv"
    file_test = "/home/yi/sentimentAnalysis/data/train_test_split/tripadvisor/test_tripadvisor.csv"

    five_score_train_df = pd.read_csv(file_train, index_col=0)
    five_score_test_df = pd.read_csv(file_test, index_col=0)

    names = ["5_score_test.csv", "5_score_train.csv"]

    for name, df in zip(names, [five_score_test_df, five_score_train_df]):
        df.drop(["_id", "date", "hotelLocation", "hotelName", "hotelStars", "hotelUrl", "title", "url", "userId"], axis=1, inplace=True)
        df.to_csv("/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/"+name)



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


