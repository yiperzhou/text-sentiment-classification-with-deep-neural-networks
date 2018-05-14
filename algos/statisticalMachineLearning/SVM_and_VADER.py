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
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
import pandas as pd
import csv
import datetime
from sklearn.metrics import accuracy_score
import sys
import liblinearutil

def clf_VADER(x_test):
    '''
    reviews sentiment classification on VADER package
    :param x_test:
    :return:
    '''
    start = time.time()
    y_pred = list()
    analyzer = SentimentIntensityAnalyzer()
    for i in x_test:
        vs = analyzer.polarity_scores(i)
        # map [-1,1] to [1,5], mapping rule is below:
        # [-1, -0.6), [-0.6, -0.2), [-0.2, 0.2), [0.2, 0.6), [0.6, 1)
        score = vs["compound"]
        if score >= -1 and score < -0.6:
            y_pred.append(1)
        elif score >= -0.6 and score < -0.2:
            y_pred.append(2)
        elif score >= -0.2 and score < 0.2:
            y_pred.append(3)
        elif score >= 0.2 and score < 0.6:
            y_pred.append(4)
        elif score >= 0.6 and score <= 1:
            y_pred.append(5)
        else:
            print("VADER score not in [-1, 1]")
            sys.exit()
    elapse = time.time() - start
    return y_pred, elapse


def clf_SVM(X_train, y_train, X_test):

    '''
    train the model with stastical machine learning methods, here is linear SVC
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    '''
    start = time.time()

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', SVC(decision_function_shape='ovo')),
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


def liblinear_svm(x_train, y_train, x_test, y_test):
    x_train = x_train[:100]
    y_train = y_train[:100]

    x_test = x_test[:100]
    y_test = y_test[:100]


    start = time.time()
    vector = TfidfVectorizer(min_df=3, max_df=0.95)
    x_train_vectors = vector.fit_transform(x_train)
    x_test_vectors = vector.fit_transform(x_test)

    print("type : ", type(x_train_vectors))
    print(x_train_vectors[0])
    print("liblinear svm")

    elapse = time.time() - start

    p_labs = 0


    # model = train(y_train, x_train)
    # y是list/tuple类型，长度为l的训练标签向量
    # x是list/tuple类型的训练实例，list中的每一个元素是list/tuple/dictory类型的feature向量

    # examples
    # y, x = svm_read_problem('../heart_scale')
    # 读入libsvm格式的数据
    prob = liblinearutil.problem(list(y_train), x_train_vectors)
    # 将y,x写作prob
    param = liblinearutil.parameter('-s 3 -c 5 -q')
    # 将参数命令写作param

    # m = train(y, x, '-c 5')
    # m = train(prob, '-w1 5 -c 5')
    m = liblinearutil.train(prob, param)
    # 进行训练
    #
    # CV_ACC = train(y, x, '-v 3')
    # # -v 3 是指进行3-fold的交叉验证
    # # 返回的是交叉验证的准确率
    #
    # best_C, best_rate = train(y, x, '-C -s 0')
    #
    p_labs, p_acc, p_vals = liblinearutil.predict(list(y_test), x_test_vectors, m)
    # # y是testing data的真实标签，用于计算准确率
    # # x是待预测样本
    # # p_labs: 预测出来的标签
    # # p_acc: tuple类型，包括准确率，MSE，Squared correlation coefficient(平方相关系数)
    # # p_vals: list, 直接由模型计算出来的值，没有转化成1，0的值，也可以看做是概率估计值
    #
    (ACC, MSE, SCC) = liblinearutil.evaluations(p_labs, list(y_test))
    # # ty: list, 真实值
    # # pv: list, 估计值
    print("finish prediction")
    return p_labs, elapse

if __name__ == "__main__":

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    prefix_path = "./" + os.sep + ts_str

    train_filePath = "/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/5_score_train.csv"
    test_filePath = "/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/5_score_test.csv"

    train_data = pd.read_csv(train_filePath)
    X_train = train_data["review"]
    y_train = train_data["score"]

    # check the consistent size of reviews and sentiment
    assert len(X_train) == len(y_train)


    test_data = pd.read_csv(test_filePath)
    X_test = test_data["review"]
    y_test = test_data["score"]

    assert len(X_test) == len(y_test)



    print("train data size : ", len(y_train), "test data size : ", len(y_test))






    # SVM prediction
    y_pred_SVM, time_SVM = liblinear_svm(X_train, y_train, X_test,y_test)

    # print("multiclass liblinear SVM: ", accuracy_score(y_test, y_pred_SVM))
    # SVM_predictions_csv = np.column_stack((X_test, y_pred_SVM))
    # svm_out_path = prefix_path + os.sep + "liblinear_SVM_prediction.csv"
    # with open(svm_out_path, 'w') as f:
    #     csv.writer(f).writerows(SVM_predictions_csv)
    # f.close()




    # # first test on VADER system
    # y_pred_VADER, time_VADER = clf_VADER(X_test)
    # print("VADER elapsed time: ", round(time_VADER, 2), " s")
    #
    # # os.makedirs(prefix_path)
    #
    # # Save the evaluation to a csv
    # VADER_predictions_csv= np.column_stack((X_test, y_pred_VADER))
    #
    # vader_out_path = "VADER_prediction.csv"
    # with open(vader_out_path, 'w') as f:
    #     csv.writer(f).writerows(VADER_predictions_csv)
    # f.close()


    # # find these reviews which are wrongly classified
    # X_test = list(X_test)
    # y_test = list(y_test)
    # # wrong_clf_reviews_VADER = dict()
    #
    # wrong_clf_reviews_list = list()
    # print("test size length: ", len(y_test))
    #
    # assert len(y_test) == len(y_pred_VADER)
    #
    # for i in range(len(y_pred_VADER)):
    #     if y_pred_VADER[i] != y_test[i]:
    #         wrong_clf_reviews_list.append([y_pred_VADER[i], y_test[i], i, X_test[i], "VADER"])
    #     else:
    #         pass
    #
    # # Print and plot the confusion matrix
    # cm_VADER = metrics.confusion_matrix(y_test, y_pred_VADER)
    #
    # print("VADER metrics report: ")
    # print(metrics.classification_report(y_test, y_pred_VADER, target_names=None))
    # print("VADER confusion matrix: ")
    # print(cm_VADER)
    #
    # # second test on SVM
    # y_pred_SVM, time_SVM = clf_SVM()
    # print("SVM elapsed time : ", round(time_SVM, 2), " s")
    #
    # assert len(y_pred_SVM) == len(y_test)
    # # wrong_clf_reviews_SVM = dict()
    # for i in range(len(y_pred_SVM)):
    #     if y_pred_SVM[i] != y_test[i]:
    #         wrong_clf_reviews_list.append([y_pred_SVM[i], y_test[i], i, X_test[i], "SVM"])
    #     else:
    #         pass
    #
    # # Print and plot the confusion matrix
    # cm_SVM = metrics.confusion_matrix(y_test, y_pred_SVM)
    # print("SVM metrics report: ")
    # print(metrics.classification_report(y_test, y_pred_SVM, target_names=None))
    # print("SVM confusion matrix: ")
    # print(cm_SVM)
    #
    # wrong_clf_reviews = pd.DataFrame(wrong_clf_reviews_list, columns=["predlabel", "trueLabel", "indexLocat", "review", "classification"])
    #
    # wrong_clf_reviews.to_csv("wrong_clf_reviews.csv")
