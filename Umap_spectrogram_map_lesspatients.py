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
import umap
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    

with tf.Graph().as_default() as eog_graph:
    session_conf1 = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
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
        best_dir1 = os.path.join(config.checkpoint_path, "best_model_acc")
        saver1.restore(sess1, best_dir1)
        print("Model loaded")            
    
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
            
            #load model to train further
            best_dir2 = os.path.join(config.checkpoint_path2, "best_model_acc")
            saver2.restore(sess2, best_dir2)
            print("Model loaded")
    
#            # initialize all variables
#            print("Model initialized")
#            sess2.run(tf.compat.v1.global_variables_initializer())
#    
    
            def dev_step(y_batch,x_batch):
                feed_dict = {
                    net.input_x: x_batch,
                    net.input_y: y_batch,
                    net.training: False
                }
                output_loss, total_loss, yhat = sess2.run(
                       [net.output_loss, net.loss, net.outputlayer], feed_dict)
                return np.sum(output_loss), total_loss, yhat
    
    
            def evaluate(gen,gen2=None):
                # Validate the model on the entire evaluation test set after each epoch
                output_loss =0
                total_loss = 0
                yhat = np.zeros([29*129, len(gen.super_generator.datalist)])
                ygt=np.zeros([29*129, len(gen.super_generator.datalist)])
                x= np.zeros([29*129, len(gen.super_generator.datalist)])
                yygt=np.zeros(len(gen.super_generator.datalist))
                yyeogmap=np.zeros(len(gen.super_generator.datalist))
                num_batch_per_epoch = len(gen)
                test_step = 0
                while test_step < num_batch_per_epoch:
                    (x_batch, y_batch) = gen[test_step]
                    (x_batch2,y_batch2) = gen2[test_step]
                    assert(np.sum(y_batch!=y_batch2)==0)
                    x_batch_eog=x_batch2[:,0,:,:,1:4]
                    x_batch_c=x_batch2[:,0,:,:,0:3]
                    x_c=x_batch[:,:,0]
                    x_eog=x_batch[:,:,1]
                    y_batch=y_batch[:,0]
                    output_loss_, total_loss_, yhat_ = dev_step(x_c, x_eog)
                    x_batch_eog[:,:,:,0]=np.reshape(yhat_,[-1,29,129])
                    output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_batch_eog,y_batch)
                    output_loss += output_loss_
                    total_loss += total_loss_
                    yhat[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(yhat_)
                    ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_c)
                    yygt[(test_step)*config.batch_size :(test_step+1)*config.batch_size] = y_batch
                    x[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_eog)
                    yyeogmap[(test_step)*config.batch_size :(test_step+1)*config.batch_size] = yhat0
                    test_step += 1
                if len(gen.super_generator.datalist) - test_step*config.batch_size==1:
                    yhat=yhat[:,0:-1]
                    ygt=ygt[:,0:-1]
                    x=x[:,0:-1]
                    yygt=yygt[0:-1]
                    yyeogmap=yyeogmap[0:-1]
                elif len(gen.super_generator.datalist) > test_step*config.batch_size:
                    (x_batch, y_batch) = gen[test_step]
                    (x_batch2,y_batch2) = gen2[test_step]
                    assert(np.sum(y_batch!=y_batch2)==0)
                    x_batch_eog=x_batch2[:,0,:,:,1:4]
                    x_batch_c=x_batch2[:,0,:,:,0:3]
                    x_c=x_batch[:,:,0]
                    x_eog=x_batch[:,:,1]
                    y_batch=y_batch[:,0]
    #                with sess.as_default():
    #                with sess1.as_default():
                    output_loss_, total_loss_, yhat_ = dev_step(x_c, x_eog)
                    x_batch_eog[:,:,:,0]=np.reshape(yhat_,[-1,29,129])
                    output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_batch_eog,y_batch)
                    yhat[:, (test_step)*config.batch_size : len(gen.super_generator.datalist)] = np.transpose(yhat_)
                    ygt[:, (test_step)*config.batch_size : len(gen.super_generator.datalist)] = np.transpose(x_c)
                    yygt[(test_step)*config.batch_size : len(gen.super_generator.datalist)] = y_batch
                    x[:, (test_step)*config.batch_size : len(gen.super_generator.datalist)] = np.transpose(x_eog)
                    yyeogmap[(test_step)*config.batch_size :len(gen.super_generator.datalist)] = yhat0
                    output_loss += output_loss_
                    total_loss += total_loss_
                print(accuracy_score(yyeogmap, yygt))
                return yhat, ygt,x, yygt, yyeogmap
    
        print('Test')
        featf, feat, x, ygt, yyeogmap  = evaluate(gen=test_generatorC34,gen2=test_generatorC340)
        print('Train')
        featf2, feat2, x2, ygt2, yyeogmap2= evaluate(gen=train_generatorC34,gen2=train_generatorC340)
        print('Eval')
        featf3, feat3, x3, ygt3, yyeogmap3 = evaluate(gen=eval_generatorC34,gen2=eval_generatorC340)
        x_eog= np.transpose(np.concatenate([x, x2, x3],1))
        feat_c= np.transpose(np.concatenate([feat, feat2, feat3],1))
        feat_cf = np.transpose(np.concatenate([featf,featf2,featf3],1))
        ygt_c=np.concatenate([ygt, ygt2, ygt3])+1
        dff_feat=np.mean(np.abs(feat_c-feat_cf))
        meansqdff=np.mean(np.power(feat_c-feat_cf,2))
        dff_featx=np.mean(np.abs(feat_c-x_eog))
        meansqdffx=np.mean(np.power(feat_c-x_eog,2))
        print(dff_feat, dff_featx)
        print(meansqdff, meansqdffx)

#        output_loss, total_loss, featf, feat, x , ygt = evaluate(gen=test_generatorC34)
#        output_loss, total_loss, featf2, feat2, x2, ygt2 = evaluate(gen=train_generatorC34)
#        output_loss, total_loss, featf3, feat3, x3, ygt3 = evaluate(gen=eval_generatorC34)
#        x_eog= np.transpose(np.concatenate([x, x2, x3],1))
#        feat_c= np.transpose(np.concatenate([feat, feat2, feat3],1))
#        feat_cf = np.transpose(np.concatenate([featf,featf2,featf3],1))
#        ygt_c= np.concatenate([ygt, ygt2, ygt3])+1
#        
#        dff_feat=np.mean(np.abs(feat_c-feat_cf))
#        meansqdff=np.mean(np.power(feat_c-feat_cf,2))
#        dff_featx=np.mean(np.abs(feat_c-x_eog))
#        meansqdffx=np.mean(np.power(feat_c-x_eog,2))
#        print(dff_feat, dff_featx)
#        print(meansqdff, meansqdffx)
        
        reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
        
        trans= reducer.fit(feat_c)
        embeddingc=trans.transform(feat_c)
        embeddingf2=trans.transform(feat_cf)
        embeddingx2=trans.transform(x_eog)
        transf= reducer.fit(feat_cf)
        embeddingf= transf.transform(feat_cf)
        embeddingc2=transf.transform(feat_c)
        transe= reducer.fit(x_eog)
        embeddingx= transe.transform(x_eog)
        embeddingf3= transe.transform(feat_cf)
        embeddingc3= transe.transform(feat_c)
        
        #embedding_t= reducer.fit_transform(feat_t)
        
#        #colors_age=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_c]
#        colors_age=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_c]
#        #colors_aget=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_t]
#        colors_aget=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_t]
        
        colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_c]
#        colorst=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_t]
        #sns.palplot(sns.color_palette('hls',7))
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingf2[:,0], embeddingf2[:,1],color=colors, s=.1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingx2[:,0], embeddingx2[:,1],color=colors, s=.1)

        fig=plt.figure()
        ax=fig.add_subplot(111)        
        ax.scatter(embeddingf[:,0], embeddingf[:,1],color=colors, s=.1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingc2[:,0], embeddingc2[:,1],color=colors, s=.1)

        fig=plt.figure()
        ax=fig.add_subplot(111)        
        ax.scatter(embeddingx[:,0], embeddingx[:,1],color=colors, s=.1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingc3[:,0], embeddingc3[:,1],color=colors, s=.1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(embeddingf3[:,0], embeddingf3[:,1],color=colors, s=.1)


