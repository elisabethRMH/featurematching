#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping feature representation of EOG (of network trained on EOG) 
to feature representation of C34 (of network trained on C34)

Created on Tue Oct  6 16:37:54 2020

@author: ehereman
"""
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
from scipy.io import loadmat, savemat

#from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


sys.path.insert(1,'/users/sista/ehereman/Documents/code/adversarial_DA/')
from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config
from adversarialnetwork_featureinput import AdversarialNetworkF

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

from mappingnetwork import MappingNetwork

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_adversarialV2 import SubGenFromFile

filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"
#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
#filename='data_split_eval.mat'
#filename='data_split_eval_SS1-3.mat' # MORE data to train, same test and eval set

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch2'; # no overlap
#source='/users/sista/ehereman/Desktop/all_data_epoch4'; # no overlap
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap
normalize=True
#source='/users/sista/ehereman/Documents/code/fold3_eval_data'
#root = '/esat/biomeddata/ehereman/MASS_toolbox'
test_files=files_folds['test_sub']#[fold][0][0]
eval_files=files_folds['eval_sub']#[fold][0][0]
train_files=files_folds['train_sub']#[fold][0][0]

config= Config()

train_generatorC34= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize_per_subject=normalize) #TODO adapt back
#train_generatorEOG= train_generatorC34.copy()
#train_generatorEOG.switch_channels(channel_to_use=1)

test_generatorC34 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize_per_subject=normalize)
#test_generatorEOG= test_generatorC34.copy()
#test_generatorEOG.switch_channels(channel_to_use=1)

eval_generatorC34= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize_per_subject=normalize)
#eval_generatorEOG= eval_generatorC34.copy()
#eval_generatorEOG.switch_channels(channel_to_use=1)

train_batches_per_epoch = np.floor(len(train_generatorC34)).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generatorC34)).astype(np.uint32)
test_batches_per_epoch = np.floor(len(test_generatorC34)).astype(np.uint32)

#E: nb of epochs in each set (in the sense of little half second windows)
print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generatorC34._indices), len(eval_generatorC34._indices), len(test_generatorC34._indices)))

#E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_subjnorm'
#config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_only1pat'
config.out_dir1 = '/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_adaptednetwork2_map10pat'
config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_adaptc34network2_subjnorm_map10pat/group0'
config.checkpoint_dir= './checkpoint/'
config.allow_soft_placement=True
config.log_device_placement=False
config.nchannel=1
config.domainclassifierstage2=False
config.add_classifierinput=False
config.out_path1= config.out_dir1 #os.path.join(config.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
config.out_path = config.out_dir#os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(0.0))
config.domainclassifier = False  
config.training_epoch = 40
config.adapt_featextractor=True
config.cycle=True
config.cycle_weight=1.0
#config.evaluate_every= 100
#config.learning_rate= 1E-5
            
config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_dir1,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.out_path1)): os.makedirs(os.path.abspath(config.out_path1))
if not os.path.isdir(os.path.abspath(config.checkpoint_path1)): os.makedirs(os.path.abspath(config.checkpoint_path1))

config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))

config.checkpoint_path2 = os.path.abspath(os.path.join(config.out_dir2,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.checkpoint_path2)): os.makedirs(os.path.abspath(config.checkpoint_path2))

       
        
with tf.Graph().as_default() as c34_graph:
    session_conf = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=c34_graph, config=session_conf)
    with sess.as_default():
        arnnC34=AdversarialNetwork(config, session=sess)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(arnnC34.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
        saver = tf.train.Saver(tf.all_variables())
        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(config.checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")

with tf.Graph().as_default() as map_graph:
    session_conf2 = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    session_conf2.gpu_options.allow_growth = True
    sess2 = tf.Session(graph=map_graph, config=session_conf2)
    with sess2.as_default():
        net= MappingNetwork(config, session=sess2)
        
        # for batch normalization
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step2 = tf.Variable(0, name="global_step", trainable=False)
            optimizer2 = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
            grads_and_vars2 = optimizer2.compute_gradients(net.loss)
            train_op2 = optimizer2.apply_gradients(grads_and_vars2, global_step=global_step2)
            
        #TODO vanaf hier!!!!

        print("Writing to {}\n".format(config.out_dir2))

        saver2 = tf.compat.v1.train.Saver(tf.all_variables(), max_to_keep=1)
        #sess.run(tf.initialize_all_variables())
        best_dir2 = os.path.join(config.checkpoint_path2, 'best_model_acc')#"best_model_acc")
        saver.restore(sess2, best_dir2)
        print("Model loaded")

with tf.Graph().as_default() as c34fi_graph:
    session_conf1 = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    session_conf1.gpu_options.allow_growth = True
    sess1 = tf.Session(graph=c34fi_graph, config=session_conf1)
    with sess1.as_default():
        arnnC34_FI=AdversarialNetworkF(config, session=sess1)

        # Define Training procedure
        global_step1 = tf.Variable(0, name="global_step", trainable=False)
        optimizer1 = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars1 = optimizer1.compute_gradients(arnnC34_FI.loss)
        train_op1 = optimizer1.apply_gradients(grads_and_vars1, global_step=global_step1)
        
        # Load saved model to continue training or initialize all variables
        saver1 = tf.compat.v1.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess1.run(tf.compat.v1.global_variables_initializer())
 
    
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              arnnC34_FI.feat_input: x_batch,
              arnnC34_FI.input_y: y_batch
            }
            _, step, output_loss, total_loss = sess1.run(
               [train_op1, global_step1, arnnC34_FI.output_loss_mean, arnnC34_FI.loss],
               feed_dict)
            return step, output_loss, total_loss

        def dev_step(x_batch, y_batch):
            feed_dict = {
                arnnC34_FI.feat_input: x_batch,
                arnnC34_FI.input_y: y_batch,
            }
            output_loss, total_loss, yhat = sess1.run(
                   [arnnC34_FI.output_loss_mean, arnnC34_FI.loss, arnnC34_FI.predictionC], feed_dict)
            return output_loss, total_loss, yhat

#        def evalfeatures_adversarialnets(arnn, x_batch, y_batch):
#            frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
#            feed_dict = {
#                arnn.input_x: x_batch,
#                arnn.input_y: y_batch,
#                arnn.dropout_keep_prob_rnn: 1.0,
#                arnn.frame_seq_len: frame_seq_len
#            }
#            output_loss, domain_loss, total_loss, yhat, score, features = sess.run(
#                   [arnn.output_loss_mean,arnn.domain_loss_mean, arnn.loss, arnn.predictionC, arnn.score_C, arnn.attention_out1], feed_dict)
#            return output_loss, total_loss, yhat, score, features

        def evaluate(gen, log_filename):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss =0
            total_loss = 0
            yhat = np.zeros(len(gen.datalist))
            num_batch_per_epoch = len(gen)
            test_step = 0
            ygt = np.zeros(len(gen.datalist))
            while test_step < num_batch_per_epoch:
                (x_batch, y_batch) = gen[test_step]
                x_c=x_batch[:,0,:,:,0:3]
                x_eog=x_batch[:,0,:,:,1:4]
                y_batch=y_batch[:,0]
                output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnC34.evalfeatures_adversarialnets( x_eog, y_batch)
                output_loss, total_loss,x_batch_eog, features=net.evaluate(x_batch_eog, x_batch_c)
                
#                    x_batch=np.moveaxis(x_batch,2,-1)                
#                    x_batch=np.moveaxis(x_batch,2,3)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch_eog, y_batch) #in case of eog orig, you have to take x_batch_eog as input
                output_loss += output_loss_
                total_loss += total_loss_
                yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
                ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                test_step += 1
            if len(gen.datalist) - test_step*config.batch_size==1:
                yhat=yhat[0:-1]
                ygt=ygt[0:-1]
                
            elif len(gen.datalist) > test_step*config.batch_size:
                (x_batch, y_batch) = gen[test_step]
                x_c=x_batch[:,0,:,:,0:3]
                x_eog=x_batch[:,0,:,:,1:4]
                y_batch=y_batch[:,0]
                output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnC34.evalfeatures_adversarialnets( x_eog, y_batch)
                output_loss, total_loss,x_batch_eog, features=net.evaluate(x_batch_eog, x_batch_c) #only add this line for training class layer on mapped eog, not for training it on original eog

                output_loss_, total_loss_, yhat_ = dev_step(x_batch_eog, y_batch) #in case of eog orig, you have to take x_batch_eog as input
                yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
                ygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1
            ygt= ygt+1
            acc = accuracy_score(ygt, yhat)
            with open(os.path.join(config.out_dir1, log_filename), "a") as text_file:
                text_file.write("{:g} {:g}, acc: {:g}\n".format(output_loss, total_loss, acc))
            return acc, yhat, output_loss, total_loss

        start_time = time.time()
        # Loop over number of epochs
        time_lst=[]
        best_acc=0.0
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 0
            while step < train_batches_per_epoch:
                # Get a batch
                t1=time.time()
                (x_batch, y_batch) = train_generatorC34[step]
                x_c=x_batch[:,0,:,:,0:3]
                x_eog=x_batch[:,0,:,:,1:4]
                y_batch=y_batch[:,0]
                #with sess.as_default():
                output_loss, total_loss, yhat, score, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                #with sess1.as_default():
                output_loss, total_loss, yhat, score, x_batch_eog = arnnC34.evalfeatures_adversarialnets(x_eog,y_batch)
                output_loss, total_loss, x_batch_eog, features=net.evaluate(x_batch_eog, x_batch_c) #only add this line for training class layer on mapped eog, not for training it on original eog
                t2=time.time()
                train_step_, train_output_loss_, train_total_loss_ = train_step(x_batch_eog, y_batch) #features -> x_batch_eog if on eog orig
                
                time_lst.append(t2-t1)
#                    x_batch=np.moveaxis(x_batch,2,-1)
#                    x_batch=np.moveaxis(x_batch,2,3)
                time_str = datetime.now().isoformat()


                print("{}: step {}, output_loss {}, total_loss {} ".format(time_str, train_step_, train_output_loss_, train_total_loss_))
                step += 1

                current_step = tf.compat.v1.train.global_step(sess1, global_step1)
                if current_step % config.evaluate_every == 0:
                    # Validate the model on the entire evaluation test set after each epoch
                    print("{} Start validation".format(datetime.now()))
                    eval_acc, yhat, output_loss, total_loss = evaluate(gen=eval_generatorC34, log_filename="eval_result_log.txt")
                    acc, yhat, output_loss, total_loss = evaluate(gen=test_generatorC34, log_filename="test_result_log.txt")

                    if(eval_acc >= best_acc):
                        best_acc = eval_acc
                        checkpoint_name = os.path.join(config.checkpoint_path1, 'model_step' + str(current_step) +'.ckpt')
                        save_path = saver1.save(sess1, checkpoint_name)

                        print("Best model updated")
                        print(checkpoint_name)
                        source_file = checkpoint_name
                        dest_file = os.path.join(config.checkpoint_path1, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')


                    #test_generator.reset_pointer()
                    #eval_generator.reset_pointer()
            #train_generator.reset_pointer()
            train_generatorC34.on_epoch_end()

        end_time = time.time()
        with open(os.path.join(config.out_dir1, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
            text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
        save_neuralnetworkinfo(config.checkpoint_path1, 'classificationnetwork',arnnC34_FI,  originpath=__file__, readme_text=
                'Classification layer (normalization per patient), \n eog features from adapted c34 network (posttrainingmap_cycle_adaptc34network2_subjnorm_map10pat), trained on 10 patients \n \n'+
                print_instance_attributes(config))
