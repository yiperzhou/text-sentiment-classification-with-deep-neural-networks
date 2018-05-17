import tensorflow as tf
import numpy as np
import os
import datetime
import time
from rnn import RNN
from tensorflow.contrib import learn
import data_helpers
import sys
from sklearn.model_selection import train_test_split
import itertools
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train/test data (Default: 100)")
tf.flags.DEFINE_string("data_file", "/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/5_score_train.csv", "data source file")

# Model Hyperparameters
tf.flags.DEFINE_bool("enable_word_embeddings", True, "Enable/disable the word embedding, default: True, using Google word2vec")
tf.flags.DEFINE_string("cell_type", "lstm", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 100, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


# default glove word2vec dimension
# embedding_dimension = 100
glove_embedding_dimension = 50
glove_num_filters = 50
word2vec_num_filters = 300
word2vec_embedding_dimension = 300

glove_hidden_size = 50
word2vec_hidden_size = 300


x_text, y = data_helpers.load_data(filePath=FLAGS.data_file)
max_document_length = max([len(x.split(" ")) for x in x_text])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x_train, x_dev, y_train, y_dev = train_test_split(x_text, y, test_size=0.2)

vocab_processor.fit_transform(x_text)

del x_text, y



print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

def train():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=max_document_length,
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=glove_embedding_dimension,
                cell_type=FLAGS.cell_type,
                hidden_size=glove_hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.enable_word_embeddings:
                # initial matrix with random uniform
                # initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.embedding_dim))
                initW = None
                vocabulary = vocab_processor.vocabulary_

                glove_file = '../glove/glove.6B.50d.txt'
                word2vec_file = '../googleWord2Vec/GoogleNews-vectors-negative300.txt'

                # load any vectors from the word2vec
                # print("Load word2vec file {0}".format(FLAGS.word2vec))
                print("load google word2vec file ......")

                initW = data_helpers.load_embedding_vectors_glove(vocabulary, glove_file, glove_embedding_dimension)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained glove model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...

            # Generate dev batches
            dev_batches = data_helpers.batch_iter(
                list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)


            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(list(vocab_processor.transform(x_batch)))
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    top10_dev_batches = itertools.islice(dev_batches, 0, 10, 1)
                    for dev_batch in top10_dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        x_dev_batch = np.array(list(vocab_processor.transform(x_dev_batch)))
                        feed_dict_dev = {
                            rnn.input_text: x_dev_batch,
                            rnn.input_y: y_dev_batch,
                            rnn.dropout_keep_prob: 1.0
                        }
                        summaries_dev, loss, accuracy = sess.run(
                            [dev_summary_op, rnn.loss, rnn.accuracy], feed_dict_dev)
                        dev_summary_writer.add_summary(summaries_dev, step)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))


                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
