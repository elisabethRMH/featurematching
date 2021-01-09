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
from datagenerator.subgenerators.subgenlambda import SubGenLambda

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

from mappingnetworkspectro import MappingNetwork_spectro

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

for number_patients in [41]:
    
    test_files=files_folds['test_sub']#[fold][0][0]
    eval_files=files_folds['eval_sub']#[fold][0][0]
    train_files=files_folds['train_sub'][:,0:number_patients]#[fold][0][0]
    
    config= Config()
    config.l2_reg_lambda=0.02
    
    train_generatorC340= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=normalize) #TODO adapt back
    train_generatorC34=SubGenLambda(train_generatorC340, lambda x,y: (np.reshape(x[:,:,:,:,0:2],[-1, 129*29,2]), y), 'all')
    #train_generatorEOG= train_generatorC34.copy()
    #train_generatorEOG.switch_channels(channel_to_use=1)
    
    test_generatorC340 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=normalize)
    test_generatorC34=SubGenLambda(test_generatorC340, lambda x,y: (np.reshape(x[:,:,:,:,0:2],[-1, 129*29,2]), y), 'all')
    #test_generatorEOG= test_generatorC34.copy()
    #test_generatorEOG.switch_channels(channel_to_use=1)
    
    eval_generatorC340= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=normalize)
    eval_generatorC34=SubGenLambda(eval_generatorC340, lambda x,y: (np.reshape(x[:,:,:,:,0:2],[-1, 129*29,2]), y), 'all')
    #eval_generatorEOG= eval_generatorC34.copy()
    #eval_generatorEOG.switch_channels(channel_to_use=1)
    
    train_batches_per_epoch = np.floor(len(train_generatorC34)).astype(np.uint32)
    eval_batches_per_epoch = np.floor(len(eval_generatorC34)).astype(np.uint32)
    test_batches_per_epoch = np.floor(len(test_generatorC34)).astype(np.uint32)
    
    #E: nb of epochs in each set (in the sense of little half second windows)
    print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generatorC340._indices), len(eval_generatorC340._indices), len(test_generatorC340._indices)))
    
    #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
    print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
    
    config.out_dir1 = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum'#_only{:d}pat'.format(number_patients)
    config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum'
#    config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_EOGtrain{:d}pat'.format(number_patients)
    config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/spectromap_eogtoc34_diffnet2_map{:d}pat'.format(number_patients)
    #config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_linearmap'
    config.checkpoint_dir= './checkpoint/'
    config.allow_soft_placement=True
    config.log_device_placement=False
    config.nchannel=1
    config.domainclassifierstage2=False
    config.add_classifierinput=False
    config.out_path1= os.path.join(config.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
    config.out_path = os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(0.0))
    config.domainclassifier = False  
    config.evaluate_every= 200 #int(100*number_patients*2/40)
#    config.evaluate_every=100
    config.learning_rate= 1E-5
    config.training_epoch=25#int(25*10/number_patients)#25
#    config.training_epoch=25
                
    config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_path1,config.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(config.out_path1)): os.makedirs(os.path.abspath(config.out_path1))
    if not os.path.isdir(os.path.abspath(config.checkpoint_path1)): os.makedirs(os.path.abspath(config.checkpoint_path1))
    
    config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
    if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
    
    config.checkpoint_path2 = os.path.abspath(os.path.join(config.out_dir2,config.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(config.checkpoint_path2)): os.makedirs(os.path.abspath(config.checkpoint_path2))
    

            
    
    with tf.Graph().as_default() as map_graph:
        session_conf2 = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        session_conf2.gpu_options.allow_growth = True
        sess2 = tf.Session(graph=map_graph, config=session_conf2)
        with sess2.as_default():
            net= MappingNetwork_spectro(config, session=sess2)
            
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
            
#            #load model to train further
#            best_dir2 = os.path.join(config.checkpoint_path2, "best_model_acc")
#            saver2.restore(sess2, best_dir2)
#            print("Model loaded")
    
            # initialize all variables
            print("Model initialized")
            sess2.run(tf.compat.v1.global_variables_initializer())
    
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  net.input_x: x_batch,
                  net.input_y: y_batch,
                  net.training:True
                }
                _, step, output_loss, total_loss = sess2.run(
                   [train_op2, global_step2, net.output_loss_mean, net.loss],
                   feed_dict)
                return step, output_loss, total_loss
    
            def dev_step(y_batch,x_batch):
                feed_dict = {
                    net.input_x: x_batch,
                    net.input_y: y_batch,
                    net.training:False
                }
                output_loss, total_loss, yhat = sess2.run(
                       [net.output_loss, net.loss, net.outputlayer], feed_dict)
                return np.sum(output_loss), total_loss, yhat
    
    
            def evaluate(gen, log_filename):
                # Validate the model on the entire evaluation test set after each epoch
                output_loss =0
                total_loss = 0
                yhat = np.zeros([29*129, len(gen.super_generator.datalist)])
                num_batch_per_epoch = len(gen)
                test_step = 0
                while test_step < num_batch_per_epoch:
                    (x_batch, y_batch) = gen[test_step]
                    x_c=x_batch[:,:,0]
                    x_eog=x_batch[:,:,1]
                    output_loss_, total_loss_, yhat_ = dev_step(x_c, x_eog)
                    output_loss += output_loss_
                    total_loss += total_loss_
                    yhat[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(yhat_)
                    test_step += 1
                if len(gen.super_generator.datalist) - test_step*config.batch_size==1:
                    yhat=yhat[:,0:-1]
                    
                elif len(gen.super_generator.datalist) > test_step*config.batch_size:
                    (x_batch, y_batch) = gen[test_step]
                    x_c=x_batch[:,:,0]
                    x_eog=x_batch[:,:,1]
    #                with sess.as_default():
    #                with sess1.as_default():
                    output_loss_, total_loss_, yhat_ = dev_step(x_c, x_eog)
                    yhat[:, (test_step)*config.batch_size : len(gen.super_generator.datalist)] = np.transpose(yhat_)
                    output_loss += output_loss_
                    total_loss += total_loss_
                with open(os.path.join(config.out_dir2, log_filename), "a") as text_file:
                    text_file.write("{:g} {:g} {:g}\n".format(output_loss, total_loss, output_loss/len(gen.super_generator.datalist)))
                return output_loss, total_loss
    
            start_time = time.time()
            # Loop over number of epochs
            best_loss=np.inf
            for epoch in range(config.training_epoch):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                step = 0
                while step < train_batches_per_epoch:
                    # Get a batch
                    (x_batch, y_batch) = train_generatorC34[step]
                    x_c=x_batch[:,:,0]
                    x_eog=x_batch[:,:,1]
                    #with sess.as_default():
                    
    #                    x_batch=np.moveaxis(x_batch,2,-1)
    #                    x_batch=np.moveaxis(x_batch,2,3)
                    train_step_, train_output_loss_, train_total_loss_ = train_step(x_c, x_eog)
                    time_str = datetime.now().isoformat()
    
    
                    print("{}: step {}, output_loss {}, total_loss {} ".format(time_str, train_step_, train_output_loss_, train_total_loss_))
                    step += 1
    
                    current_step = tf.compat.v1.train.global_step(sess2, global_step2)
                    if current_step % config.evaluate_every == 0:
                        # Validate the model on the entire evaluation test set after each epoch
                        print("{} Start validation".format(datetime.now()))
                        output_loss, total_loss = evaluate(gen=eval_generatorC34, log_filename="eval_result_log.txt")
                        output_loss, total_loss = evaluate(gen=test_generatorC34, log_filename="test_result_log.txt")
    
                        if(output_loss <= best_loss):
                            best_loss = output_loss
                            checkpoint_name = os.path.join(config.checkpoint_path2, 'model_step' + str(current_step) +'.ckpt')
                            save_path = saver2.save(sess2, checkpoint_name)
    
                            print("Best model updated")
                            print(checkpoint_name)
                            source_file = checkpoint_name
                            dest_file = os.path.join(config.checkpoint_path2, 'best_model_acc')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')
    
    
                        #test_generator.reset_pointer()
                        #eval_generator.reset_pointer()
                #train_generator.reset_pointer()
                train_generatorC34.on_epoch_end()
    
