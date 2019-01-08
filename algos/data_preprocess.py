import spacy
from torchtext import data
from tqdm import tqdm
import pandas as pd
import torch
from torchtext.vocab import Vectors, GloVe
from torch.nn import init

spacy_en = spacy.load('en')


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
        csv_data = csv_data.head(1000)

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

    # original dataset

    # # narvi
    # train_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_train_dataset.csv"
    # test_path = "/home/zhouy/thesis/data/text_classification_data/tripadvisor_test_dataset.csv"
    

    # tut thinkstation
    train_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/tripadvisor_train_dataset.csv"
    test_path = "/media/yi/harddrive/codes/thesis_sentimentAnalysis/data/text_classification_data/tripadvisor_test_dataset.csv"


    # # xps
    # train_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/train.csv"
    # test_path = "D:/sentimentAnalysis/algos/5_ToxicCommentClassification-pytorch/data/test.csv"

    # tripadvisor dataset
    ## xps
    #train_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_train_dataset.csv"
    #test_path = "D:/sentimentAnalysis/data/text_classification_data/tripadvisor_test_dataset.csv"

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

    TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    

    test = CustomDataset(test_path, text_field=TEXT, label_field=LABEL, test=True)
    
    train = CustomDataset(train_path, text_field=TEXT, label_field=LABEL)
    # should save the above train, test, these two variables.




    vectors = GloVe(name='6B', dim=args.embed_dim)
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

    net = Model(args)
    # copy pretrained glove word embedding into the model
    net.embedding.weight.data.copy_(TEXT.vocab.vectors)
    if using_gpu:
        net.cuda()

    return train_iter, test_iter, net
