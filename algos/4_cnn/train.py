#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sys
from sklearn.model_selection import train_test_split
import itertools
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/5_score_train.csv", "data source file")

# Model Hyperparameters
tf.flags.DEFINE_bool("enable_word_embeddings", True, "Enable/disable the word embedding, default: True, using Google word2vec")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.load_data(filePath=FLAGS.data_file)

# default glove word2vec dimension
# embedding_dimension = 100
glove_embedding_dimension = 50
glove_num_filters = 50
word2vec_num_filters = 300
word2vec_embedding_dimension = 300

# Build vocabulary
# 找到一条评论中，单词数量最多的那个
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#fit the vocab from glove
# pretrain = vocab_processor.fit(vocab)
#transform inputs

#return batch dataset
# def batch_iter(x, y, batch_size=64):
#
#     #get dataset and label
#     data_size=len(x)
#     num_batches_per_epoch=int((data_size-1)/batch_size)
#     for batch_index in range(num_batches_per_epoch):
#         start_index=batch_index*batch_size
#         end_index=min((batch_index+1)*batch_size,data_size)
#         return_x = x[start_index:end_index]
#         return_y = y[start_index:end_index]
#
#         yield (return_x,return_y)
#
# for step, (x, y) in enumerate(batch_iter(x_test, y)):
#     # 这里的x变成了一个word index的二维数组
#     x = np.asarray(list(vocab_processor.fit_transform(x)))
vocab_processor.fit_transform(x_text)



# Randomly shuffle data
# np.random.seed(10)
# # shuffle_indices 是一个一维数组， 实际上重点关注的是应该是x[shuffle_indices], 可以这样操作，也就是传入一个一维数组作为下标
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = np.array(x_text)[shuffle_indices]
# y_shuffled = np.array(y)[shuffle_indices]
#
# # Split train/test set
# # TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]




x_train, x_dev, y_train, y_dev = train_test_split(x_text, y, test_size=0.2)

del x_text, y

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_document_length,
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=word2vec_embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=word2vec_num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
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
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        if FLAGS.enable_word_embeddings:
            vocabulary = vocab_processor.vocabulary_
            initW = None

            glove_file = '../glove/glove.6B.50d.txt'
            word2vec_file = '../googleWord2Vec/GoogleNews-vectors-negative300.txt'

            # load glove pretrain word embedding
            print("load glove file ......")
            # initW = data_helpers.load_embedding_vectors_glove(vocabulary, glove_file, embedding_dimension)

            initW = data_helpers.load_embedding_word2vec(vocabulary, word2vec_file, word2vec_embedding_dimension)
            print("glove file has been loaded")
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Generate dev batches
        dev_batches = data_helpers.batch_iter(
            list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
        # only evaluate first 10 batch dev dataset
        top10_dev_batches = itertools.islice(dev_batches, 0, 10,1)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)

            x_batch = np.array(list(vocab_processor.transform(x_batch)))
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")


                for dev_batch in top10_dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    x_dev_batch = np.array(list(vocab_processor.transform(x_dev_batch)))
                    dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)

                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
