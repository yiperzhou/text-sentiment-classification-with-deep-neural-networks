import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from random import randint
import re
import json
import os
import csv
import datetime
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
# import liblinearutil
from opts import args
from helper import accuracy, AverageMeter, log_stats, LOG, plot_figs, save_misclassified_reviews, confusion_matrix
from data_preprocess import prepare_data_svm

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
    global logFile
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
        temp_result = str(i) + "params - " + str(grid_search.cv_results_['params'][i]) + "; mean - " + str(grid_search.cv_results_['mean_test_score'][i]) + "; std - " + str(grid_search.cv_results_['std_test_score'][i])
        LOG(temp_result, logFile)


    y_pred = grid_search.predict(X_test)
    elapse = time.time() - start
    return y_pred, elapse


# def liblinear_svm(x_train, y_train, x_test, y_test):
#     # x_train = x_train[:100]
#     # y_train = y_train[:100]
#     #
#     # x_test = x_test[:100]
#     # y_test = y_test[:100]


#     start = time.time()
#     vector = TfidfVectorizer(min_df=3, max_df=0.95)
#     x_train_vectors = vector.fit_transform(x_train)
#     x_test_vectors = vector.fit_transform(x_test)

#     print("type : ", type(x_train_vectors))
#     print(x_train_vectors[0])
#     print("liblinear svm")



#     p_labs = 0


#     # model = train(y_train, x_train)
#     # y是list/tuple类型，长度为l的训练标签向量
#     # x是list/tuple类型的训练实例，list中的每一个元素是list/tuple/dictory类型的feature向量

#     # examples
#     # y, x = svm_read_problem('../heart_scale')
#     # 读入libsvm格式的数据
#     prob = liblinearutil.problem(list(y_train), x_train_vectors)
#     # 将y,x写作prob

#     ss = [3]
#     cs = [1, 2, 3, 4]
#     es = [0.1, 0.2, 0.3]
#     grid_result = []
#     for s in ss:
#         for c in cs:
#             for e in es:
#                 parameter = '-s ' + str(s) + ' -c ' + str(c) + ' -e ' + str(e) + ' -v 5'
#                 param = liblinearutil.parameter(parameter)
#                 m = liblinearutil.train(prob, param)
#                 p_labs, p_acc, p_vals = liblinearutil.predict(list(y_test), x_test_vectors, m)
#                 (ACC, MSE, SCC) = liblinearutil.evaluations(p_labs, list(y_test))
#                 grid_result.append([s, c, e, ACC])

#     print(grid_result)


#     p_labs, p_acc, p_vals = liblinearutil.predict(list(y_test), x_test_vectors, m)
#     # # y是testing data的真实标签，用于计算准确率
#     # # x是待预测样本
#     # # p_labs: 预测出来的标签
#     # # p_acc: tuple类型，包括准确率，MSE，Squared correlation coefficient(平方相关系数)
#     # # p_vals: list, 直接由模型计算出来的值，没有转化成1，0的值，也可以看做是概率估计值
#     #
#     (ACC, MSE, SCC) = liblinearutil.evaluations(p_labs, list(y_test))
#     # # ty: list, 真实值
#     # # pv: list, 估计值
#     elapse = time.time() - start

#     print("finish prediction, prediction time: ", elapse)
#     return p_labs, elapse





def main(**kwargs):
    global args

    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # program_start_time = time.time()
    instanceName = "classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__))

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model + os.sep + ts_str+"_"+args.dataset

    os.makedirs(path)
    
    args.savedir = path

    global logFile
    logFile = path + os.sep + "log.txt"

    LOG(str(args), logFile)

    X_train, y_train, X_test, y_test = prepare_data_svm(args)
    
    LOG("train data size : " + str(len(y_train)) + " test data size : " + str(len(y_test)), logFile)

    if args.model == "liblinear_svm":
        # # liblinear_svm prediction
        # y_pred_SVM, time_SVM = liblinear_svm(X_train, y_train, X_test, y_test)
        # LOG("time elapse: " + str(time_SVM), logFile)
        # LOG("multiclass liblinear SVM: " + str(accuracy_score(y_test, y_pred_SVM)), logFile)
        # SVM_predictions_csv = np.column_stack((X_test, y_pred_SVM))

        # SVM_predictions_csv.to_csv(path + os.sep + "test_classification_result.csv", sep=',', index=True)
        pass

    elif args.model == "svm" or args.model == "SVM":
        # SVM prediction
        y_pred, time = clf_SVM(X_train, y_train, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        LOG("time elapse: " + str(time) + " seconds", logFile)
        LOG("[SVM] accuracy: " + str(accuracy), logFile)

        df = pd.DataFrame(data={"test review": X_test,
                                "test_label": y_pred,
                                "ground truth": y_test})
        df.to_csv(path + os.sep + "test_classification_result.csv", sep=',', index=True)

    else:
        NotImplementedError

    LOG("============Finish============", logFile)
    
    # svm_out_path ="liblinear_SVM_prediction_4rd_run.csv"
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

    #calculate confusion matrix 
    logFile = confusion_matrix(y_pred, y_test, logFile)
    
    # save misclassified reviews
    wrong_clf_reviews = save_misclassified_reviews(X_test, y_pred, y_test, args.model)

    wrong_clf_reviews.to_csv(path + os.sep + "wrong_clf_reviews.csv", sep=',', index=True)



    # # second test on SVM
    # y_pred_SVM, time_SVM = clf_SVM()
    # print("SVM elapsed time : ", round(time_SVM, 2), " s")
    
    # assert len(y_pred_SVM) == len(y_test)
    # # wrong_clf_reviews_SVM = dict()
    # for i in range(len(y_pred_SVM)):
    #     if y_pred_SVM[i] != y_test[i]:
    #         wrong_clf_reviews_list.append([y_pred_SVM[i], y_test[i], i, X_test[i], "SVM"])
    #     else:
    #         pass
    # #
    # # # Print and plot the confusion matrix
    # # cm_SVM = metrics.confusion_matrix(y_test, y_pred_SVM)
    # # print("SVM metrics report: ")
    # # print(metrics.classification_report(y_test, y_pred_SVM, target_names=None))
    # # print("SVM confusion matrix: ")
    # # print(cm_SVM)
    # #
    # # wrong_clf_reviews = pd.DataFrame(wrong_clf_reviews_list, columns=["predlabel", "trueLabel", "indexLocat", "review", "classification"])
    # #
    # # wrong_clf_reviews.to_csv("wrong_clf_reviews.csv")


if __name__ == "__main__":
    # call SVM text classification algorithm
    main()

    # calcualte confusion matrix
    # vdcnn_text_clf_result_csv = pd.read_csv("D:\sentimentAnalysis\algos\Classification_Accuracy\VDCNN\2019-01-21-14-33-42_tripadvisor\test_classification_result.csv")
    # confusion_matrix(y_pred, y_test, logFile)
