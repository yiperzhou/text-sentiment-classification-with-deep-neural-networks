import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import time
import datetime
from opts import args

import data_preprocess
from nets import models
from nets import CNN_Text_Model
from helper import accuracy, AverageMeter, log_stats


def main(**kwargs):
    global args

    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    print(args)

    program_start_time = time.time()
    instanceName = "Classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__))

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model + os.sep + ts_str

    os.makedirs(path)

    if args.model == "BiLSTMConv":
        Model = models.BiLSTMConv

    elif args.model == "BiGRU":
        Model = models.BiGRU

    elif args.model == "WordCNN":
        Model = models.WordCNN

    elif args.model == "BiGRUWithTimeDropout":
        Model = models.BiGRUWithTimeDropout

    elif args.model == "CNN_Text_Model":
        Model = CNN_Text_Model.CNN_Text

    else:
        NotImplementedError

    # process the input data.
    

    train_iter, test_iter, net = data_preprocess.prepare_data_and_model(Model=Model, args=args, using_gpu=True)

    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=.99)
    criterion = nn.CrossEntropyLoss().cuda()

    best_test_acc = 0
    best_test_results = []

    epoch_train_accs = []
    epoch_train_losses = []
    epoch_test_accs = []
    epoch_test_losses = []
    epoch_lrs = []

    for epoch in range(args.epochs):

        train_accs = []
        train_losses = []

        for batch in tqdm(train_iter):
            lr_scheduler.step()

            net.train()
            xs = batch.text
            ys = batch.label

            ys = ys.cuda(async=True)
            xs = torch.autograd.Variable(xs)
            ys_var = torch.autograd.Variable(ys)

            logits = net(xs)
            loss = criterion(logits, ys_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item()/int(args.batch_size))
            train_accs.append(accuracy(logits.data, ys))

        print("epoch {} :  training accumulated accuracy {} %".format(epoch, np.mean(train_accs)))
        epoch_train_accs.append(np.mean(train_accs))
        epoch_train_losses.append(np.mean(train_losses))

        torch.save(net.state_dict(), "{}.pkl".format(Model.name))

        test_accs = []
        test_losses = []
        net.eval()

        results = []
        print("running testing.....")
        for batch in tqdm(test_iter):
            xs_test = batch.text
            ys_test = batch.label

            logits_test = net(xs_test)
            test_loss = criterion(logits_test, ys_test)

            test_losses.append(test_loss.item() / int(args.batch_size))
            test_accs.append(accuracy(logits_test.data, ys_test))

            results = results + logits_test.topk(1, 1, True, True)[1].t().cpu().numpy().tolist()[0]

        print("epoch {} :  testing accumulated accuracy {} %".format(epoch, np.mean(test_accs)))

        epoch_test_accs.append(np.mean(test_accs))
        epoch_test_losses.append(np.mean(test_losses))

        if best_test_acc < np.mean(test_accs):
            best_test_acc = np.mean(test_accs)
            best_test_results = results

        # epoch_lrs.append(0.1)


        log_stats(path, [np.mean(train_accs)], [np.mean(train_losses)], [np.mean(test_accs)], [np.mean(test_losses)])

    df = pd.DataFrame(data={"test_label": best_test_results})
    df.to_csv(path + os.sep + "test_classification_result.csv", sep=',', index=True)




if __name__ == '__main__':
    main()