import pandas as pd
import os
import numpy as np


folder = "2018-05-11-01-12-53"
file1 = folder + os.sep + "CNN_no_pretrain_word_embedding_prediction.csv"
file2 = folder + os.sep + "CNN_glove_50dims_prediction.csv"
file3 = folder + os.sep + "SVM_prediction.csv"
file4 = folder + os.sep + "VADER_prediction.csv"


pd1 = pd.read_csv(file1, header=None, names=["review", "sentiment_1"])
pd2 = pd.read_csv(file2, header=None)
pd3 = pd.read_csv(file3, header=None)
pd4 = pd.read_csv(file4, header=None)


sentiment_2 = pd2.iloc[:, -1]
sentiment_3 = pd3.iloc[:, -1]
sentiment_4 = pd4.iloc[:, -1]
pd1["sentiment_2"] = sentiment_2
pd1["sentiment_3"] = sentiment_3
pd1["sentiment_4"] = sentiment_4


std_list = []
for index, row in pd1.iterrows():
    dev = np.std([row["sentiment_1"], row["sentiment_2"], row["sentiment_3"], row["sentiment_4"]])
    std_list.append(dev)

pd1["deviation"] = std_list

pd1.to_csv("std_deviation_test_tripadvisor_4_algos.csv")


