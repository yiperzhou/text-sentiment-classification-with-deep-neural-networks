import tensorflow as tf
import numpy as np
import os
import time
from rnn_model import RNN_Model
import data_helper
import pandas as pd
from sklearn import metrics
import csv

flags =tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.1,'the learning rate')
flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
flags.DEFINE_integer('vocabulary_size',20000,'vocabulary_size')
flags.DEFINE_integer('emdedding_dim',128,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',128,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
flags.DEFINE_integer('max_len',40,'max_len of training sentence')
flags.DEFINE_integer('valid_num',100,'epoch num of validation')
flags.DEFINE_integer('checkpoint_num',1000,'epoch num of checkpoint')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('class_num',2,'class num')
flags.DEFINE_float('keep_prob',0.5,'dropout rate')
flags.DEFINE_integer('num_epoch',1,'num epoch')
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')
flags.DEFINE_integer('check_point_every',10,'checkpoint every num epoch ')


class Config(object):

    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=FLAGS.vocabulary_size
    embed_dim=FLAGS.emdedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    class_num=FLAGS.class_num
    keep_prob=FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size=FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num=FLAGS.valid_num
    out_dir=FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every


def evaluate(model,session,data,global_steps=None,summary_writer=None):


    correct_num=0
    predict_result = list()
    total_num=len(data[0])
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=FLAGS.batch_size)):

         fetches = model.correct_num
        #  predict_detail = model.prediction
        #  predict = list()
         feed_dict={}
         feed_dict[model.input_data]=x
         feed_dict[model.target]=y
         feed_dict[model.mask_x]=mask_x
         # model.assign_new_batch_size(session,len(x))
         state = session.run(model._initial_state)
         for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
         count=session.run(fetches,feed_dict)
        #  predict = session.run()
        #  session.run(model.logits)
        #  session.run(tf.Print(model.correct_prediction))
         correct_num+=count
         predict_result.append(session.run(model.prediction, feed_dict))
        
    print("total num:", total_num)
    print("total predict num:", len(predict_result))
    accuracy=float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy',accuracy)

    dev_summary = session.run(dev_summary)
    # dev_predict_detail = tf.summary.scalar('dev_predict_detail', predict_detail)
    if summary_writer:
        summary_writer.add_summary(dev_summary,global_steps)
        summary_writer.flush()
    return accuracy, predict_result

def run_epoch(model,session,data,global_steps,valid_model,valid_data,train_summary_writer,valid_summary_writer=None):
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=FLAGS.batch_size)):

        feed_dict={}
        feed_dict[model.input_data]=x
        feed_dict[model.target]=y
        feed_dict[model.mask_x]=mask_x
        # model.assign_new_batch_size(session,len(x))
        fetches = [model.cost,model.accuracy,model.train_op,model.summary]
        state = session.run(model._initial_state)
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        cost,accuracy,_,summary = session.run(fetches,feed_dict)
        train_summary_writer.add_summary(summary,global_steps)
        train_summary_writer.flush()
        valid_accuracy, predictResult =evaluate(valid_model,session,valid_data,global_steps,valid_summary_writer)
        if(global_steps%100==0):
            print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f"%(global_steps,cost,accuracy,valid_accuracy))
        global_steps+=1

    return global_steps





def train_step():

    print("loading the dataset...")
    config = Config()
    eval_config=Config()
    eval_config.keep_prob=1.0

    train_data,valid_data,test_data=data_helper.load_data(FLAGS.max_len,batch_size=config.batch_size)

    print("begin training")

    # gpu_config=tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,1*FLAGS.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = RNN_Model(config=config,is_training=True)

        with tf.variable_scope("model",reuse=True,initializer=initializer):
            valid_model = RNN_Model(config=eval_config,is_training=False)
            test_model = RNN_Model(config=eval_config,is_training=False)

        #add summary
        train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        train_summary_writer =  tf.summary.FileWriter(train_summary_dir,session.graph)

        dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir,session.graph)

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())


        tf.global_variables_initializer().run()
        global_steps=1
        begin_time=int(time.time())

        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(session,config.lr*lr_decay)
            global_steps=run_epoch(model,session,train_data,global_steps,valid_model,valid_data,train_summary_writer,dev_summary_writer)

            if i% config.checkpoint_every==0:
                path = saver.save(session,checkpoint_prefix,global_steps)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        test_accuracy, test_predict_result=evaluate(test_model,session,test_data)
        test_predict_result = np.concatenate(test_predict_result).ravel()

        test_file_path = "../../data/csv/test_tripadvisor_5cities.csv"
        test_df = pd.read_csv(test_file_path)
        X_test = test_df["review"]

        print("size of predict result: ", len(test_predict_result))
        print("size of original test result: ", len(X_test))

        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(X_test), test_predict_result))

        result_file_file = "rnn_prediction.csv"
        with open(result_file_file, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)



        print("the test data accuracy is %f"%test_accuracy)

        print(len(test_predict_result))
        # Print and plot the confusion matrix
        print("LSTM confusion matrix: ")
        cm_LSTM = metrics.confusion_matrix(test_data[1][:1792], test_predict_result)
        print(cm_LSTM)
        

        print("LSTM metrics report: ")
        print(metrics.classification_report(test_data[1][:1792], test_predict_result, target_names=None))

        # print("test prediction result: ", test_predict_result)

        # load the original test text review
        text_test_file = "/home/yi/sentimentAnalysis/data/csv/test_tripadvisor_5cities.csv"
        text_testDF = pd.read_csv(text_test_file)
        
        # open the previous wrong classification csv file
        # wrong_classification_csv = pd.read_csv("/home/yi/sentimentAnalysis/data-preprocess/sentiment_CLF/wrong_clf_reviews.csv")
        wrong_clf_reviews_list = list()
        for i in range(len(test_predict_result)):
            if test_data[1][i] != test_predict_result[i]:
                # just check, the review is exact same review
                if text_testDF.iloc[i]["sentiment"] == test_data[1][i]:
                    # wrong_classification_csv.append({"predlabel":test_predict_result[i], "trueLabel":test_data[1][i], "indexLocat":i, "review":text_testDF.iloc[i]["review"], "classification":"LSTM"},  ignore_index=True)
                    wrong_clf_reviews_list.append([test_predict_result[i], test_data[1][i], i, text_testDF.iloc[i]["review"], "LSTM"])
            else:
                pass

        wrong_clf_reviews = pd.DataFrame(wrong_clf_reviews_list, columns=["predlabel", "trueLabel", "indexLocat", "review", "classification"])
        wrong_clf_reviews.to_csv("wrong_clf_reviews_LSTM.csv")
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()






