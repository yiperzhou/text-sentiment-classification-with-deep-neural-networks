
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

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
        res.append(100-correct_k.mul_(100.0 / batch_size))
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


def log_stats(path, epochs_acc_train, epochs_loss_train, epochs_acc_test, epochs_loss_test):
    with open(path + os.sep + "train_acc.txt", "a") as fp:
        for a in epochs_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        for loss in epochs_loss_train:
            fp.write("%.4f " % loss)
        fp.write("\n")

    # with open(path + os.sep + "epochs_lr.txt", "a") as fp:
    #     fp.write("%.7f " % epochs_lr)
    #     fp.write("\n")

    with open(path + os.sep + "test_acc.txt", "a") as fp:
        for a in epochs_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "test_losses.txt", "a") as fp:
        for loss in epochs_loss_test:
            fp.write("%.4f " % loss)
        fp.write("\n")

def plot_figs(epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses, args, captionStrDict):
    
    """
    plot epoch test error after model testing is finished
    """

    all_y_labels = ["train error (%)", "train loss", "test error (%)", "test loss"]
    save_file_names = ["train_error.png","train_loss.png","test_error.png","test_loss.png"]
    fig_titles = [args.model + " Train Classification error"+captionStrDict["fig_title"], args.model + " Train Loss"+captionStrDict["fig_title"], args.model + " Test Classification error"+captionStrDict["fig_title"], args.model + " Test Loss"+captionStrDict["fig_title"]]
    all_stats = [epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses]
    for y_label, file_name, fig_title, data in zip(all_y_labels, save_file_names, fig_titles, all_stats):

        fig, ax0 = plt.subplots(1, sharex=True)
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