# coding: utf-8

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
import json
import os
import pandas as pd


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

    y_pred = grid_search.predict(X_test)
    elapse = time.time() - start
    return y_pred, elapse


if __name__ == "__main__":
    
    filePath = "/home/yi/sentimentAnalysis/data-preprocess/data/tripadvisor_5cities.csv"
    data = pd.read_csv(filePath)
    X = data["review"]
    y = data["sentiment"]
    
    assert len(X) == len(y)
    print("check the consistent size of reviews and sentiment : ", "review size : ", len(X), "sentiment size: ", len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    train_pf = pd.DataFrame()
    train_pf["review"] = X_train
    train_pf["sentiment"] = y_train

    test_df = pd.DataFrame()
    test_df["review"] = X_test
    test_df["sentiment"] = y_test
    if not os.path.exists("train_tripadvisor_5cities.csv"):
        train_pf.to_csv("train_tripadvisor_5cities.csv")
    if not os.path.exists("test_tripadvisor_5cities.csv"):
        test_df.to_csv("test_tripadvisor_5cities.csv")


    print(" X train shape for train data : ", X_train.shape)
    print("=================================================================================")


    # first test on VADER system
    y_pred_VADER, time_VADER = clf_VADER()
    print("VADER elapsed time: ", round(time_VADER, 2), " s")
    


    # find these reviews which are wrongly classified
    X_test = list(X_test)
    y_test = list(y_test)
    # wrong_clf_reviews_VADER = dict()
    
    wrong_clf_reviews_list = list()
    print("test size length: ", len(y_test))

    assert len(y_test) == len(y_pred_VADER)

    for i in range(len(y_pred_VADER)):
        if y_pred_VADER[i] != y_test[i]:
            wrong_clf_reviews_list.append([y_pred_VADER[i], y_test[i], i, X_test[i], "VADER"])
        else:
            pass

    # Print and plot the confusion matrix
    cm_VADER = metrics.confusion_matrix(y_test, y_pred_VADER)
    
    print("VADER metrics report: ")
    print(metrics.classification_report(y_test, y_pred_VADER, target_names=None))
    print("VADER confusion matrix: ")
    print(cm_VADER)

    # second test on SVM
    y_pred_SVM, time_SVM = clf_SVM()
    print("SVM elapsed time : ", round(time_SVM, 2), " s")

    assert len(y_pred_SVM) == len(y_test)
    # wrong_clf_reviews_SVM = dict()
    for i in range(len(y_pred_SVM)):
        if y_pred_SVM[i] != y_test[i]:
            wrong_clf_reviews_list.append([y_pred_SVM[i], y_test[i], i, X_test[i], "SVM"])
        else:
            pass

    # Print and plot the confusion matrix
    cm_SVM = metrics.confusion_matrix(y_test, y_pred_SVM)
    print("SVM metrics report: ")
    print(metrics.classification_report(y_test, y_pred_SVM, target_names=None))
    print("SVM confusion matrix: ")
    print(cm_SVM)

    wrong_clf_reviews = pd.DataFrame(wrong_clf_reviews_list, columns=["predlabel", "trueLabel", "indexLocat", "review", "classification"])

    wrong_clf_reviews.to_csv("wrong_clf_reviews.csv")
