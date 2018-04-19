# coding: utf-8

from pymongo import MongoClient
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import time
from random import randint
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def getCollection(collet = ""):
    '''
    return pandas dataframe.
    '''
    cursor = db[collet].find({})
    df = pd.DataFrame(list(cursor))
    return df


def label_sentiment_category(row):
    """
    split reviews into two categories, pos, labled by "1", neg, labeled by "0"
    [0, 1, 2] is neg, [4,5] is pos, 
    [3] is neutral, it needs to be deleted since we only do two-class classification
    """
    if row["score"] in [0, 1, 2]:
        return 0
    if row["score"] in [3]:
        return -1
    if row["score"] in [4, 5]:
        return 1

def clean_punc_and_marks(row):
    """
    remove punctuation, including ???????
    """
    words = nltk.word_tokenize(row["review"])

    words=[word.lower() for word in words if word.isalpha()]
    words = words[:250]
    return " ".join(words)

def cleanSentences(string):
    """
    removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    """
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def clf_VADER():
    """
    reviews sentiment classification on VADER package
    """
    start = time.time()
    y_pred = list()
    analyzer = SentimentIntensityAnalyzer()
    for i in X_test:
        vs = analyzer.polarity_scores(i)
        if vs["compound"] > 0:
            y_pred.append(1)
        if vs["compound"] < 0:
            y_pred.append(0)
# if the predicted review is neutral, then set it as positive since we only deal with two class classification
        if vs["compound"] == 0:
             y_pred.append(1)
    elapse = time.time() - start
    return y_pred, elapse


def clf_SVM():
    """
    train the model with stastical machine learning methods, here is linear SVC
    """
    start = time.time()

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)]}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # TASK: print the mean and std for each candidate along with the parameter
    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (grid_search.cv_results_['params'][i],
                    grid_search.cv_results_['mean_test_score'][i],
                    grid_search.cv_results_['std_test_score'][i]))

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_pred = grid_search.predict(X_test)

    # Print the classification report
    # print(metrics.classification_report(y_test, y_predicted, target_names=None))
    elapse = time.time() - start
    # print("elapsed time: ", round(elapsed_time, 2), " seconds")
    # Print and plot the confusion matrix
    # cm = metrics.confusion_matrix(y_test, y_predicted)
    # print(cm)
    # plt.matshow(cm)
    # plt.show()
    return y_pred, elapse


if __name__ == "__main__":
    # connect to the db
    client = MongoClient()
    db = client.sentimentAnalysis

    # take Barcelona city hotel reviews as example
    city = "barcelonaTripadvisor"
    data = getCollection(collet = city)

    data['sentiment'] = data.apply(lambda row: label_sentiment_category(row),axis=1)
    print("data size : ", data.shape)

    # remove row where sentiment is neutral
    data = data[data.sentiment != -1]

    print("data shape after remove neutral reviews : ", data.shape)
    data['review'] = data.apply(lambda row: clean_punc_and_marks(row),axis=1)

    X = data["review"]
    y = data["sentiment"]

    assert len(X) == len(y)
    print("check the consistent size of reviews and sentiment : ", "review size : ", len(X), "sentiment size: ", len(y))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    print(" X train shape for train data : ", X_train.shape)
    print("=================================================================================")


    # first test on VADER system
    y_pred_VADER, time_VADER = clf_VADER()
    print("VADER elapsed time: ", round(time_VADER, 2), " s")
    # Print and plot the confusion matrix
    cm_VADER = metrics.confusion_matrix(y_test, y_pred_VADER)
    
    print("VADER metrics report: ")
    print(metrics.classification_report(y_test, y_pred_VADER, target_names=None))
    print("VADER confusion matrix: ")
    print(cm_VADER)
    # plt.title("VADER sentiment classifcation")
    # plt.matshow(cm_VADER)
    # plt.show()

    # second test on SVM
    y_pred_SVM, time_SVM = clf_SVM()
    print("SVM elapsed time : ", round(time_SVM, 2), " s")
    # Print and plot the confusion matrix
    cm_SVM = metrics.confusion_matrix(y_test, y_pred_SVM)
    print("SVM metrics report: ")
    print(metrics.classification_report(y_test, y_pred_SVM, target_names=None))
    print("SVM confusion matrix: ")
    print(cm_SVM)
    # add title info
    # plt.title("SVM sentiment classification")
    # plt.matshow(cm_SVM)

    # plt.show()

    
