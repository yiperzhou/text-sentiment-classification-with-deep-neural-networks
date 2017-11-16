from pyspark.sql import SQLContext, SparkSession
from nltk.tokenize import word_tokenize
import json
import numpy as np


def countLength(strLists):
    '''
    :param strLists: list of strings
    :return: the average string length, with one decimal precision
    '''
    if len(strLists) == 0:
        return 0
    else:
        totalLen = 0
        for i in strLists:
            totalLen += len(word_tokenize(i))
        avgLen = totalLen/len(strLists)
        avgLen = ".1f".format(avgLen)
        return round(avgLen)

def word2VecGlove():
    '''
    using glove tool to train the hotel review vectors
    :return:
    '''
    return 0

# tokenize函数对review内容进行分词
def tokenize(text):
    tokens = []
#     print(text)
#     text = text.encode('ascii', 'ignore')  # to decode
#     print("-------")
#     print(text)
    text = remove_spl_char_regex.sub(" ", text)  # Remove special characters
    text = text.lower()

    for word in text.split():
        if word not in stopwords \
                and word not in string.punctuation \
                and len(word) > 1 \
                and word != '``':
            tokens.append(word)
    return tokens

def doc2vec(document):
    # 100维的向量
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
        # 查找该词在预训练的word2vec模型中的特征值
            vec = np.array(lookup_bd.value.get(word)) + 1
            # print(vec)
            # 若该特征词在预先训练好的模型中，则添加到向量中
            if vec != None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec



