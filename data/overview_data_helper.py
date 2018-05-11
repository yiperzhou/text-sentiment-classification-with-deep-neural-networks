from nltk.tokenize.regexp import WhitespaceTokenizer
import numpy as np
import sys
from datetime import datetime as dt
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



def label_sentiment_category(row):
    """
    split reviews into two categories, pos, labled by "1", neg, labeled by "0"
    [0, 1, 2] is neg, [4,5] is pos,
    [3] is neutral, it needs to be deleted since we only do two-class classification
    """
    if row["score"] in [0, 1, 2]:
        return 0
    # if row["score"] in [3]:
    #     return -1
    elif row["score"] in [3, 4, 5]:
        return 1
    else:
        print("error, review score not in [0,1,2,3,4,5]")
        sys.exit()


def str_2_date(stringDate):
    '''
    convert string date format into python comparable striptime format
    '''
    # ts = min(ts, datetime.striptime(x["date"], "%B %d, %Y").date())
    # te = max(te, datetime.striptime(x["date"], "%B %d, %Y").date())
    return dt.strptime(stringDate, "%B %d, %Y")


def avg_word_count(reviewsList):
    '''
    calculate average word count for given reviews list
    :param reviewsList:
    :return:
    '''

    reviewsLen = list()
    for review in reviewsList:
        tokens = WhitespaceTokenizer().tokenize(review)
        reviewLen = len(tokens)
        reviewsLen.append(reviewLen)

    avg_word_count = np.mean(reviewsLen)

    return avg_word_count


def date_time_analysis(datetimeList):
    '''
    get earliest date and latest date for given dataframe column
    :param datetimeList:
    :return:
    '''
    reviewsDateList = [str_2_date(dateStr) for dateStr in datetimeList]
    earliestDate = min(reviewsDateList)
    latestDate = max(reviewsDateList)

    return earliestDate, latestDate


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def topic_model_NMF(textData, n_features=1000, n_components=1, n_top_words=5):
    n_samples = len(textData)


    # Use tf-idf features for NMF.
    # print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(textData)

    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))

    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # future we can also do LDA topic modeling

    print_top_words(nmf, tfidf_feature_names, n_top_words)
