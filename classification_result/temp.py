import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# df1 = pd.read_csv("std_deviation_test_tripadvisor_4_algos.csv")
# df2 = pd.read_csv("../raw_tripadvisor_data_test.csv")
#
# df2["sentiment_1"] = df1.iloc[:, -5]
# df2["sentiment_2"] = df1.iloc[:, -4]
# df2["sentiment_3"] = df1.iloc[:, -3]
# df2["sentiment_4"] = df1.iloc[:, -2]
# df2["deviation"] = df1.iloc[:, -1]
#
#
# df2.sort_values(["deviation"], ascending=False, inplace=True)
#
#
# df2.to_csv("deviation_with_raw_test_data.csv")



df = pd.read_csv("deviation_with_raw_test_data.csv")

misclf_score = []
corclf_score = []

for index, row in df.iterrows():
    if row["deviation"] > 0:
        misclf_score.append(row["score"])
    else:
        corclf_score.append(row["score"])

# fig, ax= plt.subplot(2)
#
# ax[0] =

# tripadvisor 的评分系统是 [1,2,3,4,5], 没有2.5, 3.5 这类；
#计算全部评论的平均词长度，正确分类的词长度，错误分类的词长度

plt.hist(misclf_score)
plt.hist(corclf_score)

plt.show()