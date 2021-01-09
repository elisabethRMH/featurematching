'''
Almost identical to train_arnn_tb.py from selfsup_Banville
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat

#from arnn_sleep_sup import ARNN_Sleep
from FMandClassModel import FMandClass_Model
from fmandclassmodel_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
#from subgenfromfile_loadrandomtuples import SubGenFromFile
from subgenfromfile_fmandclassnet import SubGenFromFile

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch4'; # no overlap
source='/esat/asterie1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap



#VERSION WITH PATIENT GROUPS
for nb_patients in [1,2]:
    di2=np.load('patient_groups.npy',allow_pickle=True)
    di2=di2.item()
    for pat_group in range(len(di2[nb_patients])):
        
        pat=di2[nb_patients][pat_group]
        test_files=files_folds['test_sub']#[fold][0][0]
        eval_files=files_folds['eval_sub']#[fold][0][0]
        train_files=files_folds['train_sub']
        train_files_forEOGnet=files_folds['train_sub'][:,pat]#[fold][0][0]
        
    #    train_files_forEOGnet= train_files[:,0:nb_patients]
            
        # Parameters
        # ==================================================
        #E toevoeging
        FLAGS = tf.app.flags.FLAGS
        for attr, value in sorted(FLAGS.__flags.items()): # python3
            x= 'FLAGS.'+attr
            exec("del %s" % (x))
            
        
        # Misc Parameters
        tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
        tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        
        # My Parameters
        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum'.format(nb_patients), "Point to output directory")
        tf.app.flags.DEFINE_string("out_dir", '/esat/asterie1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork7_eog{:d}pat/group{:d}'.format(nb_patients, pat_group), "Point to output directory")
        tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
        
        tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
        
        tf.app.flags.DEFINE_integer("seq_len", 32, "Sequence length (default: 32)")
        
        tf.app.flags.DEFINE_integer("nfilter", 20, "Sequence length (default: 20)")
        
        tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
        tf.app.flags.DEFINE_integer("attention_size1", 32, "Sequence length (default: 20)")
        
        tf.app.flags.DEFINE_integer('D',100,'Number of features') #new flag!
        
        FLAGS = tf.app.flags.FLAGS
        print("\nParameters:")
        for attr, value in sorted(FLAGS.__flags.items()): # python3
            print("{}={}".format(attr.upper(), value))
        print("")
        
        # Data Preparatopn
        # ==================================================
        
        # path where some output are stored
        out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        # path where checkpoint models are stored
        checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
        if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

#        out_path1 = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir1))
        out_path1= os.path.join(FLAGS.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        # path where checkpoint models are stored
        checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
        if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
        
        config = Config()
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.training_epoch = int(60) #/6 if using load_random_tuple
        config.same_network=False
        config.feature_extractor=True
        train_generator= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=True, subjects_list2=train_files_forEOGnet) #TODO adapt back
        eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)
        config.mse_weight=1.0#0.5
        config.mse_weight=config.mse_weight*len(train_generator.datalist)/len(train_generator.datalist_subj2)
        
        
        train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator._indices), len(eval_generator._indices), len(test_generator._indices)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        
        
        
        # variable to keep track of best fscore
        best_fscore = 0.0
        best_acc = 0.0
        best_loss=np.inf
        best_kappa = 0.0
        min_loss = float("inf")
        # Training
        # ==================================================
        
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                arnn=FMandClass_Model(config)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='labelpredictor_net'),max_to_keep=1)
                if not config.same_network:
                    sess.run(tf.initialize_all_variables())
                    
                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_c34'):
                        tmp=v2.name[9:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_eog'):
                        tmp=v2.name[9:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                else:                    
                    # initialize all variables
                    print("Model initialized")
    #                sess.run(tf.initialize_all_variables())
                    best_dir = os.path.join(checkpoint_path1, "best_model_acc")
                    saver.restore(sess, best_dir)
                print("Model loaded")
        
                def train_step(x_batch, y_batch,net2bool): #not adapted
                    """
                    A single training step
                    """
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    feed_dict = {
                      arnn.net2active:net2bool,
                      arnn.input_x: x_batch,
                      arnn.input_y: y_batch,
                      arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                      arnn.frame_seq_len: frame_seq_len
                    }
                    _, step, output_loss,mse_loss, total_loss, accuracy = sess.run(
                       [train_op, global_step, arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.accuracy],
                       feed_dict)
                    return step, output_loss, mse_loss, total_loss, accuracy
        
                def dev_step(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    feed_dict = {
                        arnn.net2active:np.ones(len(x_batch)),
                        arnn.input_x: x_batch,
                        arnn.input_y: y_batch,
                        arnn.dropout_keep_prob_rnn: 1.0,
                        arnn.frame_seq_len: frame_seq_len
                    }
                    output_loss, mse_loss, total_loss, yhat = sess.run(
                           [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.prediction], feed_dict)
                    return output_loss, mse_loss, total_loss, yhat
        
                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
        
                    output_loss =0
                    total_loss = 0
                    mse_loss=0
                    yhat = np.zeros(len(gen.datalist))
                    test_step = 0
                    ygt = np.zeros(len(gen.datalist))
                    (x_batch, y_batch) = (gen.X,gen.y)
    
                    output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                    output_loss += output_loss_
                    total_loss += total_loss_
                    mse_loss+= mse_loss_
                    
                    yhat[:] = yhat_
                    ygt[:] = np.transpose(np.argmax(y_batch,axis=1))
                    test_step += 1
                    yhat = yhat + 1
                    ygt= ygt+1
                    acc = accuracy_score(ygt, yhat)
                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g}\n".format(output_loss,mse_loss, total_loss, acc))
                    return acc, yhat, output_loss, mse_loss, total_loss

#                def evaluate(gen, log_filename):
#                    # Validate the model on the entire evaluation test set after each epoch
#        
#                    output_loss =0
#                    total_loss = 0
#                    mse_loss=0
#                    yhat = np.zeros(len(gen.datalist))
#                    num_batch_per_epoch = len(gen)
#                    test_step = 0
#                    ygt = np.zeros(len(gen.datalist))
#                    while test_step < num_batch_per_epoch:
#                        #((x_batch, y_batch),_) = gen[test_step]
#                        (x_batch,y_batch,_)=gen[test_step]
#                        x_batch=x_batch[:,0]
#                        y_batch=y_batch[:,0]
#        
#                        output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
#                        output_loss += output_loss_
#                        total_loss += total_loss_
#                        mse_loss+= mse_loss_
#                        
#                        yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
#                        ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=1))
#                        test_step += 1
#                    if len(gen.datalist) - test_step*config.batch_size==1:
#                        yhat=yhat[0:-1]
#                        ygt=ygt[0:-1]
#                            
#                    elif len(gen.datalist) > test_step*config.batch_size:
#                        # if using load_random_tuple
#                        #((x_batch, y_batch),_) = gen[test_step]
#                        (x_batch,y_batch,_)=gen[test_step]
#                        x_batch=x_batch[:,0]
#                        y_batch=y_batch[:,0]
#        
#                        output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
#                        ygt[(test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=1))
#                        yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
#                        output_loss += output_loss_
#                        total_loss += total_loss_
#                        mse_loss+= mse_loss_
#                    yhat = yhat + 1
#                    ygt= ygt+1
#                    acc = accuracy_score(ygt, yhat)
#                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
#                        text_file.write("{:g} {:g} {:g} {:g}\n".format(output_loss,mse_loss, total_loss, acc))
#                    return acc, yhat, output_loss, mse_loss, total_loss
        
                # Loop over number of epochs
                for epoch in range(config.training_epoch):
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:
                        # Get a batch
                        #((x_batch, y_batch),_) = train_generator[step]
                        (x_batch,y_batch,eog_bool)=train_generator[step]
                        x_batch=x_batch[:,0]
                        y_batch=y_batch[:,0]
        
                        train_step_, train_output_loss_, train_mse_loss_, train_total_loss_, train_acc_ = train_step(x_batch, y_batch, eog_bool)
                        time_str = datetime.now().isoformat()
        
                        print("{}: step {}, output_loss {}, mse_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_mse_loss_, train_total_loss_, train_acc_))
                        step += 1
        
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            eval_acc, eval_yhat, eval_output_loss, eval_mse_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                            test_acc, test_yhat, test_output_loss, test_mse_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
        
                            if(eval_mse_loss <= best_loss):
                                best_loss = eval_mse_loss
                                checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                                save_path = saver.save(sess, checkpoint_name)
        
                                checkpoint_name1 = os.path.join(checkpoint_path, 'model_step_outputlayer' + str(current_step) +'.ckpt')
                                save_path1 = saver1.save(sess, checkpoint_name1)
        
                                print("Best model updated")
                                source_file = checkpoint_name
                                dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                shutil.copy(source_file + '.index', dest_file + '.index')
                                shutil.copy(source_file + '.meta', dest_file + '.meta')
                                
                                source_file = checkpoint_name1
                                dest_file = os.path.join(checkpoint_path, 'best_model_outputlayer_acc')
                                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                shutil.copy(source_file + '.index', dest_file + '.index')
                                shutil.copy(source_file + '.meta', dest_file + '.meta')
        
                    train_generator.on_epoch_end()
                save_neuralnetworkinfo(checkpoint_path, 'fmandclassnetwork',arnn,  originpath=__file__, readme_text=
                        'Feature matching and classification network (no normalization per patient), \n c34 net and eog net are different, both initialized with c34 net \n training on {:d} patients \n validation with mse loss instead of accuracy \n \n'.format(nb_patients)+
                        print_instance_attributes(config))
