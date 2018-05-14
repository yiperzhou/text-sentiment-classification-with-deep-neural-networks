import tensorflow as tf
from tensorflow.contrib import learn

rnn_text_vocab = "/home/yi/sentimentAnalysis/algos/rnn-text-classification-tf/runs/1526231521/text_vocab"
rnn_text_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(rnn_text_vocab)

print("rnn text vocab")


cnn_text_vocab = "/home/yi/sentimentAnalysis/algos/cnn-text-classification-tf/runs/1526159632/vocab"
cnn_text_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(cnn_text_vocab)

print("cnn text vocab")


checkpoint_file = tf.train.latest_checkpoint("/home/yi/sentimentAnalysis/algos/rnn-text-classification-tf/runs/1526221904/checkpoints")

print("checkpoint model loaded done")