import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import sys
import csv
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("data_file", "/home/yi/sentimentAnalysis/data/rev_sent_5_score_train_test/tripadvisor/5_score_test.csv", "data source file")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

x_raw, y_eval = data_helpers.load_data(FLAGS.data_file)
y_eval = np.argmax(y_eval, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
text_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_eval = np.array(list(text_vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

def eval():


    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((x_raw, all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "rnn_lstm_glove_50dims_prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()