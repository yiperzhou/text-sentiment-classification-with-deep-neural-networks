import re
import nltk

import os
import pandas as pd


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

    words = [word.lower() for word in words if word.isalpha()]
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
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


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
    return vocab, embd


def loadFasttext(filename, emb_size):
    """
    using pre-trained 300dims word embedding fasttext
    """
    vocab = []
    embd = []
    vocab.append('unk')  # 装载不认识的词
    embd.append([0] * emb_size)  # 这个emb_size = 300
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded Fasttext!')
    file.close()
    return vocab, embd

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

            with open("/home/yi/sentimentAnalysis/data/csv/tripadvisor_5cities.pickle", "wb") as f:
                pkl.dump((train_set, test_set), f)
            f.close()

            # DF = pd.DataFrame({0:reviews_index, 1:Y})
            # DF.to_pickle("/home/yi/csv-zusammenfuehren.de_r922bdrm_XY.pkl")
            print("pickle done_clean")
            return 0
    else:
        print("length error")
        return 0


if __name__ == "__main__":
    filePath = ""
    data = pd.read_csv(filePath)

    data['sentiment'] = data.apply(lambda row: label_sentiment_category(row), axis=1)
    print("data size : ", data.shape)

    # remove row where sentiment is neutral
    data = data[data.sentiment != -1]

    print("data shape after remove neutral reviews : ", data.shape)
    data['review'] = data.apply(lambda row: clean_punc_and_marks(row), axis=1)

    X = data["review"]
    y = data["sentiment"]

    assert len(X) == len(y)
    print("check the consistent size of reviews and sentiment : ", "review size : ", len(X), "sentiment size: ", len(y))

    XY = pd.DataFrame({"review": X, "sentiment": y})

    savePath = "/home/yi/sentimentAnalysis/data/csv/train_tripadvisor_5cities.csv"
    if not os.path.exists(savePath):
        XY.to_csv(savePath)

    train_file_path = "/home/yi/sentimentAnalysis/data/csv/train_tripadvisor_5cities.csv"
    test_file_path = "/home/yi/sentimentAnalysis/data/csv/test_tripadvisor_5cities.csv"

    train_df = pd.read_csv(train_file_path)
    X_train = train_df["review"]
    y_train = train_df["sentiment"]

    test_df = pd.read_csv(test_file_path)
    X_test = test_df["review"]
    y_test = test_df["sentiment"]

    train_reviewsList, test_reviewsList, reviewsSet = get_uni_words(list(X_train), list(X_test))

    # get word index for these reviews
    reviews_index(train_reviewsList, test_reviewsList, y_train, y_test, reviewsSet)