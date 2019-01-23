import spacy
from torchtext import data
from tqdm import tqdm
import pandas as pd
import torch
from torchtext.vocab import Vectors, GloVe, FastText
from torch.nn import init
import numpy as np
from torchtext.datasets.sst import SST
import torchtext.data as data
import torchtext.datasets as datasets

spacy_en = spacy.load('en')

# load SST dataset
def sst(text_field, label_field):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(64, 64, 64))
    return train_iter, dev_iter, test_iter 


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            # choose 0.25 so unknown vectors have approximately same variance as pre-trained ones
            # same as original implementation: https://github.com/yoonkim/CNN_sentence/blob/0a626a048757d5272a7e8ccede256a434a6529be/process_data.py#L95
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class CustomDataset(data.Dataset):
    # 这里是在制作自己的酒店评论数据库
    name = 'tripadvisor hotel review'
    

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        csv_data = pd.read_csv(path)
        # csv_data = csv_data.head(10000)

        print("preparing examples...")
        for i in tqdm(range(len(csv_data))):
            sample = csv_data.loc[i]
            text, label = self.process_csv_line(sample, test)
            examples.append(data.Example.fromlist([text, label], fields))

        super(CustomDataset, self).__init__(examples, fields, **kwargs)

    # @classmethod
    # def splits(cls, root='./data',
    #            train='train.csv', test='test.csv', **kwargs):
    #     return super(CustomDataset, cls).splits(
    #         root=root, train=train, test=test, **kwargs)

    def process_csv_line(self, sample, test):
        text = sample["review"]
        text = text.replace('\n', ' ')
        label = sample["score"]
        # convert label from float to int
        label = (int(label)-1)
        return text, label


def prepare_data_and_model(Model, args, using_gpu=True):

    if args.test:
        # # narvi
        train_path = "/home/zhouy/thesis/data/text_classification_data/train_try.csv"
        test_path = "/home/zhouy/thesis/data/text_classification_data/test_try.csv"
        

        # tut thinkstation
        # train_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/train_try.csv"
        # test_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/test_try.csv"

        # # tripadvisor dataset
        # # xps
        # test_path = "D:\\sentimentAnalysis\\data\\text_classification_data\\test_model_data\\rev_sent_5_score_train_test\\tripadvisor\\test_try.csv"
        # train_path = "D:\\sentimentAnalysis\\data\\text_classification_data\\test_model_data\\rev_sent_5_score_train_test\\tripadvisor\\train_try.csv"
    
    else:
        # original dataset

        # # narvi
        train_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_train_dataset.csv"
        test_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_test_dataset.csv"
        

        # # tut thinkstation
        # train_path = "/home/yi/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/train.csv"
        # test_path = "/home/yi/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/test.csv"


        # # xps
        # train_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/train.csv"
        # test_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/test.csv"

        # tripadvisor dataset
        # xps
        # train_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_train_dataset.csv"
        # test_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_test_dataset.csv"
        

    def tokenize(text):
        fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        trans_map = str.maketrans(fileters, " " * len(fileters))
        text = text.translate(trans_map)
        text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']

        tokenized_text = []
        auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s"]
        for token in text:
            if token == "n't":
                tmp = 'not'
            elif token == "'ll":
                tmp = 'will'
            elif token in auxiliary_verbs:
                tmp = 'be'
            else:
                tmp = token
            tokenized_text.append(tmp)
        return tokenized_text

    if args.dataset == 'tripadvisor':

        TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
        LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
        
        test = CustomDataset(test_path, text_field=TEXT, label_field=LABEL, test=True)
        
        train = CustomDataset(train_path, text_field=TEXT, label_field=LABEL)
        # should save the above train, test, these two variables.

        if args.wordembedding == "glove-6b":
            vectors = GloVe(name='6B', dim=args.embed_dim)
        else:
            NotImplementedError

        # # FastText
        # vectors = FastText(name='6B', dim=args.embed_dim)

        vectors.unk_init = init.xavier_uniform
        
        # 下面这行代码报错
        # TEXT.build_vocab(train, vectors=vectors, max_size=30000)

        TEXT.build_vocab(train, vectors=vectors, max_size=10000,min_freq=10)
        LABEL.build_vocab(train)
        print('train.fields', train.fields)
        print('train.name', getattr(train, 'text'))
        print('len(train)', len(train))
        print('vars(train[0])', vars(train[0]))

        # using the training corpus to create the vocabulary

        train_iter = data.Iterator(dataset=train, batch_size=args.batch_size, train=True, repeat=False, device=0 if using_gpu else -1)
        test_iter = data.Iterator(dataset=test, batch_size=args.batch_size, train=False, sort=False, device=0 if using_gpu else -1)

        # the number of unique words
        num_tokens = len(TEXT.vocab.itos)
        args.num_tokens = num_tokens

        dev_iter = test_iter

    elif args.dataset == 'SST':
        text_field = data.Field(batch_first=True, lower=True, tokenize=tokenize)
        label_field = data.Field(sequential=False, batch_first=True)
        
        train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)


        vectors = GloVe(name='6B', dim=args.embed_dim)

        text_field.build_vocab(train_data, vectors=vectors, min_freq=1)
        label_field.build_vocab(train_data)

        train_iter = data.Iterator(train_data, batch_size=args.batch_size, device= 0 if using_gpu else -1, train=True, repeat=False,
                                        sort=False, shuffle=True)
        dev_iter = data.Iterator(dev_data, batch_size=args.batch_size, device= 0 if using_gpu else -1, train=False, repeat=False,
                                        sort=False, shuffle=False)
        test_iter = data.Iterator(test_data, batch_size=args.batch_size, device= 0 if using_gpu else -1, train=False, repeat=False,
                                        sort=False, shuffle=False)

        # train_iter, dev_iter, test_iter = sst(text_field, label_field)
        # train_iter, dev_iter, test_iter = SST.iters(batch_size=16, device=0 if using_gpu else -1, vectors="glove.6B.300d")

        # config.target_class = train_iter.dataset.NUM_CLASSES
        args.num_tokens = len(text_field.vocab)
        args.num_classes = len(label_field.vocab) - 1

        print("num_classes: ", args.num_classes)
        
    if args.model == "VDCNN":
        net = Model(depth=29, vocabulary_size=args.num_tokens, embed_size=16, n_classes=args.num_classes, k=2, optional_shortcut=True)
    else:
        net = Model(args)
    # # copy pretrained glove word embedding into the model
    # net.embedding.weight.data.copy_(TEXT.vocab.vectors)
    if using_gpu:
        net.cuda()

    return train_iter, test_iter, net


def prepare_data_svm(args):

    if args.test:
        # # narvi
        # train_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_train_dataset.csv"
        # test_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_test_dataset.csv"
        

        # # tut thinkstation
        # train_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/train_try.csv"
        # test_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/test_try.csv"

        # tripadvisor dataset
        # xps
        test_path = "D:\\sentimentAnalysis\\data\\text_classification_data\\test_model_data\\rev_sent_5_score_train_test\\tripadvisor\\test_try.csv"
        train_path = "D:\\sentimentAnalysis\\data\\text_classification_data\\test_model_data\\rev_sent_5_score_train_test\\tripadvisor\\train_try.csv"
    
    else:
        # original dataset

        # # narvi
        # train_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_train_dataset.csv"
        # test_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_test_dataset.csv"
        

        # # tut thinkstation
        # train_path = "/home/yi/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/train.csv"
        # test_path = "/home/yi/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/test.csv"


        # # xps
        # train_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/train.csv"
        # test_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/test.csv"

        # tripadvisor dataset
        # xps
        train_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_train_dataset.csv"
        test_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_test_dataset.csv"
        


    train_data = pd.read_csv(train_path)
    # X_train = train_data["review"][:10000]
    # y_train = train_data["score"][:10000]
    X_train = train_data["review"]
    y_train = train_data["score"]

    # check the consistent size of reviews and sentiment
    assert len(X_train) == len(y_train)


    test_data = pd.read_csv(test_path)
    # X_test = test_data["review"][:2000]
    # y_test = test_data["score"][:2000]
    X_test = test_data["review"]
    y_test = test_data["score"]

    assert len(X_test) == len(y_test)


    return X_train, y_train, X_test, y_test
