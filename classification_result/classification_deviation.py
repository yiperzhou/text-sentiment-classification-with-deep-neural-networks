import pandas as pd
import os
import numpy as np


raw_file = "/home/yi/sentimentAnalysis/data/train_test_split/tripadvisor/test_tripadvisor.csv"

pd_raw = pd.read_csv(raw_file)



folder = "2018-05-17-21-31-22"
file1 = folder + os.sep + "CNN_glove_50dims_prediction.csv"
file2 = folder + os.sep + "CNN_word2vec_300dims_prediction.csv"
file3 = folder + os.sep + "liblinear_SVM_prediction_1st_run.csv"
file4 = folder + os.sep + "rnn_vanilla_word2vec_300dims_prediction.csv"
file5 = folder + os.sep + "VADER_prediction_5_class.csv"


pd1 = pd.read_csv(file1, header=None, names=["review", "sentiment_1"])

assert pd1.shape[0] == pd_raw.shape[0]

pd2 = pd.read_csv(file2, header=None)
pd3 = pd.read_csv(file3, header=None)
pd4 = pd.read_csv(file4, header=None)
pd5 = pd.read_csv(file5, header=None)

assert pd1.shape[0] == pd2.shape[0] == pd3.shape[0] == pd4.shape[0] == pd5.shape[0]


sentiment_1 = pd1.iloc[:, -1]
# since cnn_glove_50dims is from [0, 4]
sentiment_1 = sentiment_1 +1

sentiment_2 = pd2.iloc[:, -1]
sentiment_2 = sentiment_2 + 1

sentiment_3 = pd3.iloc[:, -1]

sentiment_4 = pd4.iloc[:, -1]
sentiment_4 = sentiment_4 + 1

sentiment_5 = pd5.iloc[:, -1]

# merge this dataframe with raw test
pd_raw["CNN_glove_50dims"] = sentiment_1
pd_raw["CNN_word2vec_300dims"] = sentiment_2
pd_raw["liblinear_SVM"] = sentiment_3
pd_raw["rnn_word2vec_300dims"] = sentiment_4
pd_raw["VADER"] = sentiment_5


std_list = []
for index, row in pd_raw.iterrows():
    dev = np.std([row["CNN_glove_50dims"], row["CNN_word2vec_300dims"], row["liblinear_SVM"], row["rnn_word2vec_300dims"], row["VADER"]])
    std_list.append(dev)

pd_raw["deviation"] = std_list

pd_raw.sort_values(by=["deviation"], ascending=False, inplace=True)

pd_raw.to_csv("std_deviation_TestDataset_tripadvisor_5_algos.csv")


