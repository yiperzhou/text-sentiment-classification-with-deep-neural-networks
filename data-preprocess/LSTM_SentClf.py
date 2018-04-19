
# coding: utf-8


# using Barcelona hotel review data and as to (5-folder sentiment analysis classifier) to train lstm classifier
import tensorflow as tf
import numpy
from pymongo import MongoClient
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.doc2vec import TaggedDocument

# using keras to implement lstm sentiment analysis
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras import optimizers
from keras.models import Sequential, Model
import multiprocessing

def getCollection(collName = ""):
    '''
    return pandas dataframe.
    '''
    cursor = db[collName].find({})
    df = pd.DataFrame(list(cursor))
    return df

# connect to mongoclient and fetch Barcelona tripadvisor
client = MongoClient()
db = client.sentimentAnalysis
cityName = "barcelonaTripadvisor"

data = getCollection(cityName)


negReviews = list()
posReviews = list()

for score, group in data.groupby("score"):
    if score in [0, 1, 2]:
        print("negtive: ", score)
        print("group len: ", len(group))
        negReviews += [i.split() for i in group["review"]]
    elif score in [4, 5]:
        print("postive: ", score)
        print("group len: ", len(group))
        posReviews += [i.split() for i in group["review"]]
    else:
        pass



testNegReviews = list()
testPosReviews = list()

for key, group in amsterdamTrip.groupby("score"):
    if key in [0, 1, 2]:
        print("negtive: ", key)
        print("group len: ", len(group))
        testNegReviews += [i.split() for i in group["review"]]
    elif key in [4, 5]:
        print("postive: ", key)
        print("group len: ", len(group))
        testPosReviews += [i.split() for i in group["review"]]
    else:
        pass

# LabeledSentence
pos_docs = list()
neg_docs = list()


for i in range(len(posReviews)):
    pos_docs.append(TaggedDocument(words=posReviews[i], tags=['TRAIN_POS_'+str(i)]))
for i in range(len(negReviews)):
    neg_docs.append(TaggedDocument(words=negReviews[i], tags=['TRAIN_NEG_'+str(i)]))    

for i in range(len(testPosReviews)):
    pos_docs.append(TaggedDocument(words=posReviews[i], tags=['TEST_POS_'+str(i)]))
for i in range(len(testNegReviews)):
    neg_docs.append(TaggedDocument(words=testNegReviews[i], tags=['TEST_NEG_'+str(i)]))


# train doc2vec classifier
model = gensim.models.Doc2Vec(neg_docs+pos_docs, min_count=1, window=10, size=100,
                              sample=1e-4, negative=5, workers=7)


model.save("./Reviews.d2v")


model = gensim.models.Doc2Vec.load('./Reviews.d2v')

for epoch in range(5):
    model.train(neg_docs+pos_docs, total_examples=model.corpus_count, epochs=model.iter)

TRAIN_SIZE = 2733

train_arrays = numpy.zeros((2733, 100))
train_labels = numpy.zeros(2733)

for i in range(2483):
    prefix_train_pos = "TRAIN_POS_" + str(i)
    train_arrays[i] = model[prefix_train_pos]
    train_labels[i] = 1
for i in range(250):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[2483 + i] = model[prefix_train_neg]
    train_labels[2483 + i] = 0


# for test hotel reviews dataset
TEST_SIZE = 2240

test_arrays = numpy.zeros((TEST_SIZE, 100))
test_labels = numpy.zeros(TEST_SIZE)

for i in range(2068):
    prefix_train_pos = "TEST_POS_" + str(i)
    test_arrays[i] = model[prefix_train_pos]
    test_labels[i] = 1
for i in range(172):
    prefix_train_neg = 'TEST_NEG_' + str(i)
    test_arrays[2068 + i] = model[prefix_train_neg]
    test_labels[2068 + i] = 0


from sklearn.linear_model import LogisticRegression
# using logistic regression as classifier 
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

classifier.score(test_arrays, test_labels)

# using svm as classifier
from sklearn import svm
svmClf = svm.SVC()
svmClf.fit(train_arrays, train_labels)

svmClf.score(test_arrays, test_labels)


# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


# below start apply lstm(long short term memory neural network)
##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, 
              nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print('Test score:', score)
    
    
    
def lstm_predict(string):
    print('loading model......')
    with open('lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    if result[0][0]==1:
        print(string,' positive')
    else:
        print(string,' negative')

# using tensorflow to implement LSTM sentiment analysis


# if score <= 2 then, the review is considered as negative, if socre >= 4, then we say these reviews are positive
model = gensim.models.Word2Vec(words)


# In[63]:


model.save('./BarcelonaGensimWord2Vec')


# In[64]:


modelBarcelona = gensim.models.Word2Vec.load("./BarcelonaGensimWord2Vec")


# In[66]:


type(modelBarcelona.wv.vocab)


# In[68]:


print(modelBarcelona.wv.vocab.keys)


# In[ ]:





# In[50]:


model.vector_size


# In[49]:


len(model.wv.vocab)


# In[52]:


model.wv.vocab


# In[55]:


model["you"]


# In[42]:


model.wv.most_similar(positive=["", "king"], negative="man")


# In[36]:


model.wv.vocab


# In[ ]:





# In[17]:


model["X"]


# In[22]:


model.wv.vocab


# In[23]:


model.accuracy


# In[24]:


model.estimate_memory


# In[10]:


model.wv["location"


# In[18]:


model.vocabulary.sample


# In[ ]:


model


# In[16]:


say_vector = model['say']


# In[10]:





# In[ ]:


stop = stopwords.words('english')


# In[16]:


#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# In[18]:


#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)

    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)

    np.save('svm_data/train_vecs.npy',train_vecs)
    print train_vecs.shape
    #Train word2vec on test tweets
    imdb_w2v.train(x_test)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print(test_vecs.shape)


# In[14]:


# Model Hyperparameters

sequence_length = 50
embedding_dim = 300        
filter_sizes = (3, 4)
num_filters = 50
dropout_prob = (0.25, 0.5)
hidden_dims = 50


# In[15]:


model = Sequential()
model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)


# In[ ]:





# In[ ]:


#  Recurrent Neural Network (RNN) using the Long Short Term Memory (LSTM) to calculate sentiment score on Barcelona
# hotel reviews
def lstm_sent(data):
    

