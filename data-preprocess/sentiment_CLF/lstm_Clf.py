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
