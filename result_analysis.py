
import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import pandas as pd
import os
import numpy as np


def deviation_overview(data):
    all_rev_nums = df.shape[0]
    dev_0_rev_nums = (df["deviation"] == 0).sum()
    dev_0_percent = (dev_0_rev_nums / all_rev_nums) * 100
    print("# of deviation = 0, ", dev_0_rev_nums, ", all # of reviews, ", all_rev_nums, ", percentage, ", dev_0_percent)

    return 0

def plot_score_distribution(data, ylabel, xlabel, legend, fig_save_path):
    score_distribution = dict()
    for score, group in data.groupby("score"):
        if score not in score_distribution.keys():
            score_distribution[score] = group.shape[0]
        else:
            score_distribution[score] = group.shape[0]

    fig, ax = plt.subplots()
    ax.bar(score_distribution.keys(), score_distribution.values())
    # ax.xticks(score_distribution.keys(), score_distribution.keys())

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    for i, v in enumerate(score_distribution.values()):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = fig_size

    plt.tight_layout()
    plt.savefig(fig_save_path)


if __name__ == "__main__":
    deviation_file = "D:\sentimentAnalysis\data\classification_result\std_deviation_TestDataset_tripadvisor_5_algos.csv"

    # dataframe schema
    # Unnamed: 0	_id	    date	hotelLocation	hotelName	        hotelStars	        hotelUrl    review
    # score	        title	url	    userId	        CNN_glove_50dims	CNN_word2vec_300dims
    # liblinear_SVM	rnn_word2vec_300dims	VADER	deviation
    df = pd.read_csv(deviation_file)

    deviation_overview(df)

    dev_0_df = df.loc[df["deviation"] == 0]
    dev_others_df = df.loc[df["deviation"] != 0]
    assert dev_0_df.shape[0] + dev_others_df.shape[0] == df.shape[0]

    path = "fig"
    # os.mkdir("fig")
    plot_score_distribution(dev_0_df, "reviews number in test dataset, deviation =0",
                            "true label on given score", "", path+os.sep+"deviation_0_review_number_distrition.pdf")

    plot_score_distribution(dev_others_df, "reviews number in test dataset, deviation !=0",
                            "given score - true label", "", path+os.sep+"deviation_not_0_review_number_distrition.pdf")




    # see the score distribution between deviation =0 and deviation != 0


    print("program done")



