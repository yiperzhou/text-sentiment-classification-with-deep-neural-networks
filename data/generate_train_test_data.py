import pandas as pd
import sys
from overview_data_helper import avg_word_count, label_sentiment_category



if __name__ == '__main__':

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
