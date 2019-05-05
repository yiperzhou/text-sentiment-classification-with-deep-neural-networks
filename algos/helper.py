import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics

def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_stats(path, epochs_acc_train, epochs_loss_train, epochs_acc_test, epochs_loss_test, epochs_lr):
    with open(path + os.sep + "train_acc.txt", "a") as fp:
        for a in epochs_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        for loss in epochs_loss_train:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "test_acc.txt", "a") as fp:
        for a in epochs_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "test_losses.txt", "a") as fp:
        for loss in epochs_loss_test:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "epochs_lr.txt", "a") as fp:
        fp.write("%.7f " % epochs_lr)
        fp.write("\n")



def plot_figs(epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses, args, captionStrDict):
    
    """
    plot epoch test error after model testing is finished
    """

    all_y_labels = ["train acc (%)", "train loss", "test acc (%)", "test loss"]
    save_file_names = ["train_acc.png","train_loss.png","test_acc.png","test_loss.png"]
    fig_titles = [args.model + " Train Classification error"+captionStrDict["fig_title"], args.model + " Train Loss"+captionStrDict["fig_title"], args.model + " Test Classification error"+captionStrDict["fig_title"], args.model + " Test Loss"+captionStrDict["fig_title"]]
    all_stats = [epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses]
    for y_label, file_name, fig_title, data in zip(all_y_labels, save_file_names, fig_titles, all_stats):

        fig, ax0 = plt.subplots()
        colormap = plt.cm.tab20

        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data))])

        # last = len(data[0])-1

        # for k in range(len(data)):
            # Plots
        x = np.arange(len(data)) + 1
        y = np.array(data)

            # if y_label in ["train loss", "test loss"] and len(data) > 1: # means model generates more than one classifier
        c_label = y_label

            # else:
                # c_label = "accuracy"

        ax0.plot(x, y, label=c_label)
        
        ax0.set_ylabel(y_label)
        ax0.set_xlabel(captionStrDict["x_label"])
        ax0.set_title(fig_title)

        ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        plt.rcParams["figure.figsize"] = fig_size
        plt.tight_layout()

        plt.savefig(args.savedir + os.sep + file_name)
        plt.close("all")  


def save_misclassified_reviews(X_test, pred_result, y_test, model_type):
    wrong_clf_reviews = []

    for i in range(len(pred_result)):
        if pred_result[i] != y_test[i]:
            wrong_clf_reviews.append([pred_result[i], y_test[i], i, X_test[i], model_type])
        else:
            pass
    
    wrong_clf_reviews = pd.DataFrame(wrong_clf_reviews, columns=["predlabel", "trueLabel", "indexLocat", "review", "algo"])

    return wrong_clf_reviews


def confusion_matrix(pred_result, ground_truth, logFile):
    cm = metrics.confusion_matrix(ground_truth, pred_result)

    LOG("metrics report: \n", logFile)
    report = metrics.classification_report(ground_truth, pred_result, target_names=None)
    LOG("\n"+ str(report), logFile)
    LOG("\n" + str(cm), logFile)
    return logFile


def calculate_deviation(pred_1_df, pred_2_df, pred_3_df, df_test):
    pred_cnn_text_model = list(pred_1_df["test_label"])
    pred_cnn_text_model = [i+1 for i in pred_cnn_text_model]
    # ground_cnn_text = list(df_cnn_text["ground truth"])

    pred_bilstm = list(pred_2_df["test_label"]) 
    pred_bilstm = [i+1 for i in pred_bilstm]

    # add 1 back since the test label is subtracted by 1 when we train the model
    
    # ground_bilstm_text = list(df_bilstm["ground truth"])

    pred_vcdnn = list(pred_3_df["test_label"])
    pred_vcdnn = [i+1 for i in pred_vcdnn]
    # ground_vcdnn_text = list(df_vcdnn["ground truth"])

    distance_list = []
    for pred_1, pred_2, pred_3, pred_ground in zip(pred_cnn_text_model, pred_bilstm, pred_vcdnn, df_test["score"]):
        # dev = np.std([row["CNN_glove_50dims"], row["CNN_word2vec_300dims"], row["liblinear_SVM"], row["rnn_word2vec_300dims"], row["VADER"]])
        distance = np.abs(pred_1-pred_ground) + np.abs(pred_2-pred_ground) + np.abs(pred_3-pred_ground)

        distance_list.append(distance)

    result_df = pd.DataFrame({
        "review": df_test["review"],
        "ground truth": df_test["score"],
        "cnn_text_model": pred_cnn_text_model,
        "bilstm_model": pred_bilstm,
        "vcdnn_model": pred_vcdnn,
        "distance": distance_list
    })


    result_df.sort_values(by=["distance"], ascending=False, inplace=True)

    return result_df