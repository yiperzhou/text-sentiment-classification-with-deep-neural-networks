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
from gensim.models.keyedvectors import KeyedVectors
import os
import tensorflow as tf
import pandas as pd
import gzip
import _pickle as pkl


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


def loadGloVe(filename):
    """
    using glove pretained word2vec, with 100 dims and 400k words
    """
    vocab = []
    embd = []
    # vocab.append('unk') #装载不认识的词
    # embd.append([0]*emb_size) #这个emb_size = 100
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd


def convert_word2vec_txt(sourceFile, destFile):
    """
    convert word2vec bin binary file into text file
    """
    model = KeyedVectors.load_word2vec_format(sourceFile, binary=True)
    model.save_word2vec_format(destFile, binary=False)


def loadWord2Vec(filename, words):
    """
    return wordembedding for words, the return dims should be [len(words) * emb_size]
    """
    vocab = []
    embd = []
    cnt = 0




    # with tf.device("/cpu:0"):
    #     input = tf.nn.embedding_lookup()

    word2vec = KeyedVectors.load_word2vec_format(filename, binary=False)
    # to get all word and then using lookup function to get these words embedding.
    wordEmbeding = word2vec.index2word(words[:5])

    print(wordEmbeding)

    print("loaded google word2vec!")
    return vocab,embd






def loadFasttext(filename, emb_size):
    """
    using pre-trained 300dims word embedding fasttext
    """
    vocab = []
    embd = []
    vocab.append('unk') #装载不认识的词
    embd.append([0]*emb_size) #这个emb_size = 300
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded Fasttext!')
    file.close()
    return vocab,embd


    return 0


def get_uni_words(trainReviews, testReviews):
    """
    get unique words for text dataset
    """
    trainReviewsList = list()
    testReviewsList = list()
    reviewsSet = set() 
    for i in range(len(trainReviews)):
        try:
            token_review = nltk.word_tokenize(trainReviews[i])

            trainReviewsList.append(token_review)
            reviewsSet |= set(token_review)
        except print(trainReviews[i]):
            return 0

    for i in range(len(testReviews)):
        try:
            token_review = nltk.word_tokenize(testReviews[i])

            testReviewsList.append(token_review)
            reviewsSet |= set(token_review)
        except print(testReviewsList[i]):
            return 0


    return trainReviewsList, testReviewsList, reviewsSet
    

def build_LSTM():
    """
    build LSTM classifier
    """
    

def reviews_index(train_reviewsList, test_reviewsList, y_train, y_test, wordsSet):
    

    print(len(train_reviewsList), len(y_train))
    print(len(test_reviewsList), len(y_test))

    wordsList = list(wordsSet)

    train_labels = list(y_train)
    test_labels = list(y_test)

    train_reviews_index = list()
    test_reviews_index = list()

    for review in train_reviewsList:
        sent_index = []
        for word in review:
            try:
                index = wordsList.index(word)
            except ValueError:
                # if word not in wordsSet, then uses 1 as its subtution
                index = 1
            sent_index.append(index)
        train_reviews_index.append(sent_index)

    for review in test_reviewsList:
        sent_index = []
        for word in review:
            try:
                index = wordsList.index(word)
            except ValueError:
                # if word not in wordsSet, then uses 1 as its subtution
                index = 1
            sent_index.append(index)
        test_reviews_index.append(sent_index)

    
    if len(train_labels) == len(train_reviews_index):
        if len(test_labels) == len(test_reviews_index):
            
            train_set = train_reviews_index, train_labels
            test_set = test_reviews_index, test_labels

            with open("/home/yi/Desktop/tripadvisor_5cities.pickle", "wb") as f:
                pkl.dump((train_set,test_set), f)
            f.close()

            # DF = pd.DataFrame({0:reviews_index, 1:Y})
            # DF.to_pickle("/home/yi/csv-zusammenfuehren.de_r922bdrm_XY.pkl")
            print("pickle done")
            return 0
    else:
        print("length error")
        return 0



if __name__ == "__main__":

    filePath = "/home/yi/Desktop/csv-zusammenfuehren.de_r922bdrm_XY.csv"
    train_file_path = "/home/yi/sentimentAnalysis/data-preprocess/sentiment_CLF/train_tripadvisor_5cities.csv"
    test_file_path = "/home/yi/sentimentAnalysis/data-preprocess/sentiment_CLF/test_tripadvisor_5cities.csv"
    # with open(filePath) as datafile:
    #     rawdata = json.load(datafile)
    data = pd.read_csv(filePath)
    X = data["review"]
    Y = data["sentiment"]

    train_df = pd.read_csv(train_file_path)
    X_train = train_df["review"]
    y_train = train_df["sentiment"]

    test_df = pd.read_csv(test_file_path)
    X_test = test_df["review"]
    y_test = test_df["sentiment"]


    assert len(X) == len(Y)
    print("check the consistent size of reviews and sentiment : ", "review size : ", len(X), "sentiment size: ", len(Y))

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=None)

    train_reviewsList, test_reviewsList, reviewsSet = get_uni_words(list(X_train), list(X_test))


    # get word index for these reviews
    reviews_index(train_reviewsList, test_reviewsList, y_train, y_test, reviewsSet)
    
    #表示最长的句子长度max_document_length
    max_document_length = max([len(review) for review in reviewsList])

    load_glove_flag = True
    if load_glove_flag:
    # glove pretrained word embedding
        gloveFileName = "/home/yi/sentimentAnalysis/data-preprocess/glove/glove.6B.50d.txt"
        start_time = time.time()
        vocab,embd = loadGloVe(gloveFileName)
        glove_elapse = time.time()-start_time
        print("glove load pretrain word embedding time : ", glove_elapse, " s")
        vocab_size = len(vocab)
        embedding_dim = len(embd[0])
        embedding = np.asarray(embd)
        print("glove")


        with tf.Session() as sess:
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            # assign operation like 
            embedding_init = W.assign(embedding_placeholder)

            sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

            from tensorflow.contrib import learn
            #init vocab processor

            vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            #fit the vocab from glove
            pretrain = vocab_processor.fit(vocab)
            #transform inputs
            input_x = np.array(list(vocab_processor.transform(reviewsList)))


            # 该代码将输入映射为词向量，但input_x为词的id
            x_embedding = tf.nn.embedding_lookup(W, input_x)


            print("tensor shape: ", x_embedding.get_shape())
            print("not done")






    load_fasttext_flag = False
    if load_fasttext_flag:
    # fasttext pretrained word embedding
        fasttextFileName = "/home/yi/sentimentAnalysis/data-preprocess/fastText/wiki.en/wiki.en.vec"
        fasttext_emb_size = 300
        start_time = time.time()
        fasttext_vocab,fasttext_embd = loadFasttext(fasttextFileName, fasttext_emb_size)
        fasttext_elapse = time.time()-start_time
        print("fasttext load pretrain word embedding time : ", fasttext_elapse, " s")
        fasttext_vocab_size = len(fasttext_vocab)
        fasttext_embedding_dim = len(fasttext_embd[0])
        fasttext_embedding = np.asarray(fasttext_embd)

        print(fasttext_vocab_size)
        print(fasttext_embedding[:20])

        print("fasttext")



    load_word2vec_flag = False
    word2vec_sourceFile = "/home/yi/sentimentAnalysis/data-preprocess/googleWord2Vec/GoogleNews-vectors-negative300.bin"
    word2vec_destFile = "/home/yi/sentimentAnalysis/data-preprocess/googleWord2Vec/GoogleNews-vectors-negative300.txt"
    if load_word2vec_flag:
        
        
        if not os.path.exists(word2vec_destFile):
            start_time = time.time()
            convert_word2vec_txt(word2vec_sourceFile, word2vec_destFile)
            convert_elapse = time.time()-start_time
            # word2vec convert from bin to txt :  653.3498952388763  s
            print("word2vec convert from bin to txt : ", convert_elapse, " s")
        
        
            print("word2vec")
        start = time.time()
        words = get_uni_words(X.tolist())
        vocab,embd = loadWord2Vec(word2vec_destFile, words)
        print("elapse time : ", tiem.time()-start, " s")

        print("not done yet")



    


