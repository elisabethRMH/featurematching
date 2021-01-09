#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cycle map:
Mapping feature representation of EOG (of network trained on EOG) 
to feature representation of C34 (of network trained on C34)
Train the mapping with less than 41 (all) patients

Train with added cycle loss: map back to original domain too and compute MSE of original compared to cycle mapped version

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


sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

from mappingnetwork2 import MappingNetwork

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_fmandclassnet import SubGenFromFile

filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"
#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
#filename='data_split_eval.mat'
#filename='data_split_eval_SS1-3.mat' # MORE data to train, same test and eval set

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch2'; # no overlap
#source='/users/sista/ehereman/Desktop/all_data_epoch4'; # no overlap
source='/esat/asterie1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap
normalize=False
#source='/users/sista/ehereman/Documents/code/fold3_eval_data'
#root = '/esat/biomeddata/ehereman/MASS_toolbox'

#VERSION WITH PATIENT GROUPS
for number_patients in [10,5,2]:
    di2=np.load('patient_groups.npy',allow_pickle=True)
    di2=di2.item()
    for pat_group in range(len(di2[number_patients])):
        
        pat=di2[number_patients][pat_group]
        test_files=files_folds['test_sub']#[fold][0][0]
        eval_files=files_folds['eval_sub']#[fold][0][0]
#        train_files=files_folds['train_sub']
#        train_files=files_folds['train_sub'][:,pat]#[fold][0][0]
        train_files=files_folds['train_sub']
        train_files_forEOGnet=files_folds['train_sub'][:,pat]#[fold][0][0]
        
                
        config= Config()
        config.batch_size=int(config.batch_size*config.batch_size/number_patients)
        
        train_generatorC34= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=normalize,subjects_list2=train_files_forEOGnet) #TODO adapt back
        train_generatorC34.normalize_per_subject()

 #VERSION4
#        boo=[idx for idx in range(len(train_generatorC34.subjects)) if train_generatorC34.subjects[idx]=='p{:d}'.format(train_files2[0][0])]
#        np.random.RandomState(None).shuffle(boo)
#        train_generatorC34.datalist=boo
        #train_generatorEOG= train_generatorC34.copy()
        #train_generatorEOG.switch_channels(channel_to_use=1)
        
        test_generatorC34 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=normalize)
        test_generatorC34.normalize_per_subject()
        #test_generatorEOG= test_generatorC34.copy()
        #test_generatorEOG.switch_channels(channel_to_use=1)
        
        eval_generatorC34= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=normalize)
        eval_generatorC34.normalize_per_subject()
        #eval_generatorEOG= eval_generatorC34.copy()
        #eval_generatorEOG.switch_channels(channel_to_use=1)
        
        train_batches_per_epoch = np.floor(len(train_generatorC34)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generatorC34)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generatorC34)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generatorC34._indices), len(eval_generatorC34._indices), len(test_generatorC34._indices)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        
#        config.out_dir1 = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum'#_only{:d}pat'.format(number_patients)
        config.out_dir1 = '/esat/asterie1/scratch/ehereman/results_transferlearning/transferlearning_c34toeog_EOGtrain{:d}pat4/group{:d}'.format(number_patients,pat_group)
        config.out_dir = '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_subjnorm'
#        config.out_dir3 = '/esat/asterie1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_samec34network3_subjnorm_map{:d}pat/group{:d}'.format(number_patients, pat_group)        
        config.out_dir2 = '/esat/asterie1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_samec34network3_unmatch_subjnorm_map{:d}pat/group{:d}'.format(number_patients, pat_group)        
#        config.out_dir2 = '/esat/asterie1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_adaptc34network2_subjnorm_map{:d}pat/group{:d}'.format(number_patients, pat_group)
        #config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_linearmap'
        config.checkpoint_dir= './checkpoint/'
        config.allow_soft_placement=True
        config.log_device_placement=False
        config.nchannel=1
        config.domainclassifierstage2=False
        config.add_classifierinput=False
        config.out_path1= os.path.join(config.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        config.out_path = config.out_dir#os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(0.0))
        config.domainclassifier = False  
        config.evaluate_every= 200#200 #int(100*number_patients*2/40)
    #    config.evaluate_every=100
        config.learning_rate= 1e-5#1E-5
        config.training_epoch=int(25*80/number_patients)#25
        config.cycle=True
        config.cycle_weight= 1.0
        config.adapt_featextractor=False
        config.densetranspose=True
        config.bothdirections=True
        config.unmatched_c34data=True
        config.withclassification=True
        
        config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_dir1,config.checkpoint_dir))#out_path1
        if not os.path.isdir(os.path.abspath(config.out_path1)): os.makedirs(os.path.abspath(config.out_path1))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path1)): os.makedirs(os.path.abspath(config.checkpoint_path1))
        
        config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
        
        config.checkpoint_path2 = os.path.abspath(os.path.join(config.out_dir2,config.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path2)): os.makedirs(os.path.abspath(config.checkpoint_path2))

#        config.checkpoint_path3 = os.path.abspath(os.path.join(config.out_dir3,config.checkpoint_dir))
#        if not os.path.isdir(os.path.abspath(config.checkpoint_path3)): os.makedirs(os.path.abspath(config.checkpoint_path3))
        
        with tf.Graph().as_default() as eog_graph:
            session_conf1 = tf.ConfigProto(
              allow_soft_placement=config.allow_soft_placement,
              log_device_placement=config.log_device_placement)
            session_conf1.gpu_options.allow_growth = True
            sess1 = tf.Session(graph=eog_graph, config=session_conf1)
            with sess1.as_default():
                arnnEOG=AdversarialNetwork(config, session=sess1)
        
                # Define Training procedure
                global_step1 = tf.Variable(0, name="global_step", trainable=False)
                optimizer1 = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars1 = optimizer1.compute_gradients(arnnEOG.loss)
                train_op1 = optimizer1.apply_gradients(grads_and_vars1, global_step=global_step1)
            
                saver1 = tf.train.Saver(tf.all_variables())
                # Load saved model to continue training or initialize all variables
                best_dir1 = os.path.join(config.checkpoint_path1, "best_model_acc")
                saver1.restore(sess1, best_dir1)
                print("Model loaded")
                
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
                
##                load model to train further
#                best_dir2 = os.path.join(config.checkpoint_path2, "best_model_acc")
#                saver2.restore(sess2, best_dir2)
#                print("Model loaded")
        
    #            # initialize all variables
                print("Model initialized")
                sess2.run(tf.compat.v1.global_variables_initializer())
#                print("Model initialized")
#                sess2.run(tf.initialize_all_variables())
#
#                varss   = tf.all_variables()
#                except_vars_mapper = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder')
#                except_vars_cycle = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_cycle')
#                varss2= [v for v in varss if v not in except_vars_mapper and v not in except_vars_cycle]
#                varss3=[v for v in except_vars_mapper or v in except_vars_cycle]
#
#                best_dir2 = os.path.join(config.checkpoint_path, "best_model_acc")
##                best_dir = os.path.join(checkpoint_path, "best_model_acc")
#                saver21 = tf.compat.v1.train.Saver(varss2, max_to_keep=1)
#                saver21.restore(sess2, best_dir2)
#                print("Model loaded")
                 
                 

#                best_dir3 = os.path.join(config.checkpoint_path3, "best_model_acc")
#                saver22 = tf.compat.v1.train.Saver(varss3, max_to_keep=1)
#                saver22.restore(sess2, best_dir3)
#                print("Model loaded")
        
                def train_step(x_batcheog, x_batchc, unmatched_y_batch, y_batch):
                    """
                    A single training step
                    """
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    feed_dict = {
                      net.input_x0: x_batcheog,
                      net.input_y: x_batchc,
                      net.label: y_batch,
                      net.unmatched_input_y: unmatched_y_batch,
                      net.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                      net.frame_seq_len: frame_seq_len,
                      net.training:True
                      }
                    _, step, output_loss, cycle_loss, cycle_loss_unmatched, class_loss, total_loss = sess2.run(
                       [train_op2, global_step2, net.output_loss_mean, net.cycle_loss_mean, net.cycle_loss_mean3, net.class_loss_mean, net.loss],
                       feed_dict)
                    return step, output_loss, cycle_loss, cycle_loss_unmatched, total_loss
        
                def dev_step(x_batcheog, x_batchc, y_batch):
                    frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                    feed_dict = {
                        net.input_x0: x_batcheog,
                        net.input_y: x_batchc,
                        net.label: y_batch,
                        net.unmatched_input_y: x_batchc, #Doesnt matter here
                        net.dropout_keep_prob_rnn: 1.0,
                        net.frame_seq_len: frame_seq_len,
                        net.training:False
                    }
                    output_loss, cycle_loss, cycle_loss_unmatched, class_loss, total_loss, acc, yhat = sess2.run(
                           [net.output_loss, net.cycle_loss, net.cycle_loss3, net.class_loss, net.loss, net.accuracy, net.outputlayer], feed_dict)
                    return np.sum(output_loss), np.sum(cycle_loss),np.sum(cycle_loss_unmatched), np.sum(class_loss), total_loss, acc, yhat
            
#                def evaluate(gen, log_filename):
#                    # Validate the model on the entire evaluation test set after each epoch
#                    output_loss =0
#                    total_loss = 0
#                    cycle_loss=0
#                    yhat = np.zeros([128, len(gen.datalist)])
#                    num_batch_per_epoch = len(gen)
#                    test_step = 0
#                    ygt = np.zeros([128, len(gen.datalist)])
#                    while test_step < num_batch_per_epoch:
#                        (x_batch, y_batch) = gen[test_step]
#                        x_c=x_batch[:,0,:,:,0:3]
#                        x_eog=x_batch[:,0,:,:,1:4]
#                        y_batch=y_batch[:,0]
#                        output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
#                        output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnC34.evalfeatures_adversarialnets( x_eog, y_batch)
#        #                    x_batch=np.moveaxis(x_batch,2,-1)                
#        #                    x_batch=np.moveaxis(x_batch,2,3)
#                        output_loss_, cycle_loss_, total_loss_, yhat_ = dev_step(x_batch_eog, x_batch_c)
#                        output_loss += output_loss_
#                        total_loss += total_loss_
#                        cycle_loss+=cycle_loss_
#                        yhat[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(yhat_)
#                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_c)
#                        test_step += 1
#                    if len(gen.datalist) - test_step*config.batch_size==1:
#                        yhat=yhat[:,0:-1]
#                        ygt=ygt[:,0:-1]
#                        
#                    elif len(gen.datalist) > test_step*config.batch_size:
#                        (x_batch, y_batch) = gen[test_step]
#                        x_c=x_batch[:,0,:,:,0:3]
#                        x_eog=x_batch[:,0,:,:,1:4]
#                        y_batch=y_batch[:,0]
#        #                with sess.as_default():
#                        output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
#                        output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnC34.evalfeatures_adversarialnets( x_eog, y_batch)
#        #                with sess1.as_default():
#                        output_loss_, cycle_loss_, total_loss_, yhat_ = dev_step(x_batch_eog, x_batch_c)
#                        yhat[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(yhat_)
#                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_c)
#                        output_loss += output_loss_
#                        cycle_loss+=cycle_loss_
#                        total_loss += total_loss_
#                    with open(os.path.join(config.out_dir2, log_filename), "a") as text_file:
#                        text_file.write("{:g} {:g} {:g} {:g}\n".format(output_loss, cycle_loss, total_loss, output_loss/len(gen.datalist)))
#                    return output_loss,cycle_loss, total_loss
                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
                    output_loss =0
                    total_loss = 0
#                    yhat = np.zeros([128, len(gen.datalist)])
                    test_step = 0
#                    ygt = np.zeros([128, len(gen.datalist)])
                    cycle_loss=0
                    (x_batch, y_batch) = (gen.X,gen.y)
                    x_c=x_batch[:,:,:,0:3]
                    x_eog=x_batch[:,:,:,1:4]
                    y_batch=np.argmax(y_batch[:],1)
                    output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                    output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnC34.evalfeatures_adversarialnets( x_eog, y_batch)
    #                    x_batch=np.moveaxis(x_batch,2,-1)                
    #                    x_batch=np.moveaxis(x_batch,2,3)
                    output_loss_, cycle_loss_, cycle_loss_unm_, class_loss, total_loss_, acc, yhat_ = dev_step(x_batch_eog, x_batch_c, y_batch) #x_eog
                    output_loss += output_loss_
                    total_loss += total_loss_
                    cycle_loss+=cycle_loss_
                    
#                    yhat = np.transpose(yhat_)
#                    ygt = np.transpose(x_batch_c)
                    test_step += 1
                    with open(os.path.join(config.out_dir2, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} {:g}\n".format(output_loss, cycle_loss, cycle_loss_unm_, class_loss, total_loss, acc, output_loss/len(gen.datalist)))
                    return output_loss, cycle_loss, total_loss
        
        
                start_time = time.time()
                # Loop over number of epochs
                time_lst=[]
                best_loss=np.inf
                for epoch in range(config.training_epoch):
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:
                        # Get a batch
                        t1=time.time()
                        (x_batch, y_batch, matched_bool) = train_generatorC34[step]
                        matched_bool=matched_bool.astype(bool)
                        x_c=x_batch[:,0,:,:,0:3]
                        x_eog=x_batch[:,0,:,:,1:4]
                        y_batch=np.argmax(y_batch[:,0],1)
                        #with sess.as_default():
                        output_loss, total_loss, yhat, score, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                        #with sess1.as_default():
                        output_loss, total_loss, yhat, score, x_batch_eog = arnnC34.evalfeatures_adversarialnets(x_eog,y_batch)
                        t2=time.time()
                        
                        time_lst.append(t2-t1)
        #                    x_batch=np.moveaxis(x_batch,2,-1)
        #                    x_batch=np.moveaxis(x_batch,2,3)
                        
                        x_batch_eog=x_batch_eog[matched_bool]
                        x_batch_c_unmatched= x_batch_c[np.invert(matched_bool)]
                        x_batch_c= x_batch_c[matched_bool]
                        y_batch_unmatched= y_batch[np.invert(matched_bool)]
        
                        train_step_, train_output_loss_, train_cycle_loss_, train_cycle_loss_unm_, train_total_loss_ = train_step(x_batch_eog, x_batch_c, x_batch_c_unmatched, y_batch_unmatched)#x_eog
                        time_str = datetime.now().isoformat()
        
        
                        print("{}: step {}, output_loss {}, cycle_loss {}, total_loss {} ".format(time_str, train_step_, train_output_loss_, train_cycle_loss_, train_total_loss_))
                        step += 1
        
                        current_step = tf.compat.v1.train.global_step(sess2, global_step2)
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            output_loss, cycle_loss, total_loss_eval = evaluate(gen=eval_generatorC34, log_filename="eval_result_log.txt")
                            output_loss, cycle_loss, total_loss = evaluate(gen=test_generatorC34, log_filename="test_result_log.txt")
        
                            if(total_loss_eval <= best_loss):
                                best_loss = total_loss_eval
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
                save_neuralnetworkinfo(config.checkpoint_path2, 'cyclemapnetwork',net,  originpath=__file__, readme_text=
                        'Cycle mapping (normalization per patient), \n unmatched C34 data added \n c34 net extracts c34 features and eog features \n tied weights \n training on {:d} patients \n \n'.format(number_patients)+
                        print_instance_attributes(config))
                
                end_time = time.time()
                with open(os.path.join(config.out_dir2, "training_time.txt"), "a") as text_file:
                    text_file.write("{:g}\n".format((end_time - start_time)))
                    text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
            