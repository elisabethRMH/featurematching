'''
Training of a feature matching model: an arnn model that learns to match features of a
modality as best as possible to corresponding features of another modality, calculated by a 
different model (that is previously trained).
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

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

#from arnn_sleep_sup import ARNN_Sleep
from FeatMatchModel import FeatMatch_Model
sys.path.insert(1,'/users/sista/ehereman/Documents/code/adversarial_DA/')
from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
#from subgenfromfile_loadrandomtuples import SubGenFromFile
from subgenfromfile_adversarialV2 import SubGenFromFile

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch4'; # no overlap
source='/esat/asterie1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap

#nb_patients= 10
#fold=0
#print('Fold: ', fold)
#    train_files=files_folds['train_sub']#[fold][0][0]
#    eval_files=files_folds['eval_sub']#[fold][0][0]
#    test_files=files_folds['test_sub']#[fold][0][0]
#    train_files= train_files[:,0:nb_patients]
for number_patients in [10,5]:
    di2=np.load('patient_groups.npy',allow_pickle=True)
    di2=di2.item()
    for pat_group in range(len(di2[number_patients])):
        
        pat=di2[number_patients][pat_group]
        test_files=files_folds['test_sub']#[fold][0][0]
        eval_files=files_folds['eval_sub']#[fold][0][0]
        train_files=files_folds['train_sub'][:,pat]#[fold][0][0]
    
    
    
    
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
        tf.app.flags.DEFINE_string("out_dir", '/esat/asterie1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_samenetwork2_eog{:d}pat/group{:d}'.format(number_patients, pat_group), "Point to output directory")
        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum'.format(number_patients), "Point to output directory")
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
        config = Config()
        
        # path where some output are stored
        out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        out_path1= os.path.join(FLAGS.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        # path where checkpoint models are stored
        checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
        if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))
    
        checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
        if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
        
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.evaluate_every= 200 #int(100*number_patients*2/40)
    #    config.evaluate_every=100
        config.learning_rate= 3E-5
        config.training_epoch=max(60,int(30*10/number_patients))#25
    #    config.training_epoch = int(60) #/6 if using load_random_tuple
        config.domainclassifier=False
        config.domainclassifierstage2=False
        config.add_classifierinput=False
        config.out_path=out_path1 #This is just to be compatible with adversarialnetwork.py (not actually being used)
        config.checkpoint_path= checkpoint_path1
        config.same_network=True
        train_generator= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=True) #TODO adapt back
        eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)
            
        
        train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator._indices), len(eval_generator._indices), len(test_generator._indices)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        
        
        
        # variable to keep track of best fscore
        best_fscore = 0.0
        best_loss=np.inf
        best_kappa = 0.0
        min_loss = float("inf")
        # Training
        # ==================================================
    
        with tf.Graph().as_default() as eog_graph:
            session_conf1 = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            session_conf1.gpu_options.allow_growth = True
            sess1 = tf.Session(graph=eog_graph, config=session_conf1)
            with sess1.as_default():
                arnnC34=AdversarialNetwork(config, session=sess1)
        
                # Define Training procedure
                global_step1 = tf.Variable(0, name="global_step", trainable=False)
                optimizer1 = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars1 = optimizer1.compute_gradients(arnnC34.loss)
                train_op1 = optimizer1.apply_gradients(grads_and_vars1, global_step=global_step1)
            
                saver1 = tf.train.Saver(tf.all_variables())
                # Load saved model to continue training or initialize all variables
                best_dir1 = os.path.join(checkpoint_path1, "best_model_acc")
                saver1.restore(sess1, best_dir1)
                print("Model loaded")
                
        
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                arnn=FeatMatch_Model(config)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        
                # initialize all variables
                print("Model initialized")
                sess.run(tf.initialize_all_variables())
                best_dir = os.path.join(checkpoint_path1, "best_model_acc")
#                best_dir = os.path.join(checkpoint_path, "best_model_acc")
                saver.restore(sess, best_dir)
                print("Model loaded")
        
                def train_step(x_batch, y_batch, featC): #not adapted
                    """
                    A single training step
                    If same_network, third input is the C34 spectrogram
                    If not same_network, third input is the C34 features calculated by C34 network
                    """
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    if config.same_network:
                        feed_dict = {
                            arnn.input_x: x_batch,
                            arnn.input_y: y_batch,
                            arnn.input_x0:featC,
                            arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                            arnn.frame_seq_len: frame_seq_len
                        }
                    else:   
                        feed_dict = {
                          arnn.input_x: x_batch,
                          arnn.input_y: y_batch,
                          arnn.features1:featC,
                          arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                          arnn.frame_seq_len: frame_seq_len
                        }
                    _, step,mse_loss, total_loss, accuracy = sess.run(
                       [train_op, global_step, arnn.mse_loss, arnn.loss, arnn.accuracy],
                       feed_dict)
                    return step, mse_loss, total_loss, accuracy
        
                def dev_step(x_batch, y_batch, featC):
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    if config.same_network:
                        feed_dict = {
                            arnn.input_x: x_batch,
                            arnn.input_y: y_batch,
                            arnn.input_x0:featC,
                            arnn.dropout_keep_prob_rnn: 1.0,
                            arnn.frame_seq_len: frame_seq_len
                        }
                    else:   
                        feed_dict = {
                            arnn.input_x: x_batch,
                            arnn.input_y: y_batch,
                            arnn.features1:featC,
                            arnn.dropout_keep_prob_rnn: 1.0,
                            arnn.frame_seq_len: frame_seq_len
                        }
                    output_loss, mse_loss, total_loss, yhat= sess.run(
                           [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.outputlayer], feed_dict)
                    return output_loss, mse_loss, total_loss,yhat
        
                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
        
                    output_loss =0
                    total_loss = 0
                    mse_loss=0
                    yhat = np.zeros(len(gen.datalist))
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt = np.zeros(len(gen.datalist))
                    while test_step < num_batch_per_epoch:
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch,y_batch)=gen[test_step]
    #                    x_batch=x_batch[:,0]
                        x_c=x_batch[:,0,:,:,0:3]
                        x_eog=x_batch[:,0,:,:,1:4]
                        y_batch=y_batch[:,0]
                        if not config.same_network:
                            output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                            output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_eog, y_batch,x_batch_c)
                        else:
                            output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_eog, y_batch,x_c)
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                        output_loss+= output_loss_
                        
                        yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
                        ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                        test_step += 1
                    if len(gen.datalist) - test_step*config.batch_size==1:
                        yhat=yhat[0:-1]
                        ygt=ygt[0:-1]
                            
                    elif len(gen.datalist) > test_step*config.batch_size:
                        (x_batch,y_batch)=gen[test_step]
                        x_c=x_batch[:,0,:,:,0:3]
                        x_eog=x_batch[:,0,:,:,1:4]
                        y_batch=y_batch[:,0]
                        if not config.same_network:
                            output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                            output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_eog, y_batch,x_batch_c)
                        else:
                            output_loss_, mse_loss_, total_loss_, yhat_ = dev_step(x_eog, y_batch,x_c)
                                    
                        
                        ygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
                        yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                        output_loss+= output_loss_
                    yhat = yhat + 1
                    ygt= ygt+1
                    acc = accuracy_score(ygt, yhat)
                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} acc {:g}\n".format(output_loss, mse_loss, total_loss, mse_loss/len(gen.datalist), acc))
                    return mse_loss, total_loss
        
                # Loop over number of epochs
                for epoch in range(config.training_epoch):
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:
                        # Get a batch
                        #((x_batch, y_batch),_) = train_generator[step]
                        (x_batch,y_batch)=train_generator[step]
    #                    x_batch=x_batch[:,0]
                        x_c=x_batch[:,0,:,:,0:3]
                        x_eog=x_batch[:,0,:,:,1:4]
                        y_batch=y_batch[:,0]
                        if not config.same_network:
                            output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                            train_step_, train_mse_loss_, train_total_loss_, acc_ = train_step(x_eog, y_batch, x_batch_c)
                        else:
                            train_step_, train_mse_loss_, train_total_loss_, acc_ = train_step(x_eog, y_batch, x_c)
                            
                        time_str = datetime.now().isoformat()
        
                        print("{}: step {}, mse_loss {}, total_loss {}, accC34 {}".format(time_str, train_step_, train_mse_loss_, train_total_loss_, acc_))
                        step += 1
        
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            eval_mse_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                            test_mse_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
        
                            if(eval_mse_loss <= best_loss):
                                best_loss = eval_mse_loss
                                checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                                save_path = saver.save(sess, checkpoint_name)
        
                                print("Best model updated")
                                source_file = checkpoint_name
                                dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                shutil.copy(source_file + '.index', dest_file + '.index')
                                shutil.copy(source_file + '.meta', dest_file + '.meta')
                                
        
                    train_generator.on_epoch_end()
                save_neuralnetworkinfo(checkpoint_path, 'featmatchnetwork',arnn,  originpath=__file__, readme_text=
                        'Feature matching (global normalization, not per patient), \n same network initialized with c34 network \n training on {:d} patients \n \n'.format(number_patients)+
                        print_instance_attributes(config))
