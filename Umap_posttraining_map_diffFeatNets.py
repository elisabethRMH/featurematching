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
import umap
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
from scipy.io import loadmat, savemat

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


sys.path.insert(1,'/users/sista/ehereman/Documents/code/adversarial_DA/')
from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config

from mappingnetwork import MappingNetwork
from adversarialnetwork_featureinput import AdversarialNetworkF

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

train_generatorC34= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=normalize) #TODO adapt back
#train_generatorEOG= train_generatorC34.copy()
#train_generatorEOG.switch_channels(channel_to_use=1)

test_generatorC34 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=normalize)
#test_generatorEOG= test_generatorC34.copy()
#test_generatorEOG.switch_channels(channel_to_use=1)

eval_generatorC34= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=normalize)
#eval_generatorEOG= eval_generatorC34.copy()
#eval_generatorEOG.switch_channels(channel_to_use=1)

train_batches_per_epoch = np.floor(len(train_generatorC34)).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generatorC34)).astype(np.uint32)
test_batches_per_epoch = np.floor(len(test_generatorC34)).astype(np.uint32)

#E: nb of epochs in each set (in the sense of little half second windows)
print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generatorC34._indices), len(eval_generatorC34._indices), len(test_generatorC34._indices)))

#E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

config.out_dir1 = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_only5pat'
config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum'
#config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network'
#config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_map10pat'
config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_2_diffnet'
config.out_dir2 = '/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_EOGtrain5pat'
config.checkpoint_dir= './checkpoint/'
config.allow_soft_placement=True
config.log_device_placement=False
config.nchannel=1
config.domainclassifierstage2=False
config.add_classifierinput=False
config.out_path1= os.path.join(config.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
config.out_path = os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(0.0))
config.domainclassifier = False  
config.evaluate_every= 100
config.learning_rate= 1E-5
            
config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_path1,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.out_path1)): os.makedirs(os.path.abspath(config.out_path1))
if not os.path.isdir(os.path.abspath(config.checkpoint_path1)): os.makedirs(os.path.abspath(config.checkpoint_path1))

config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))

config.checkpoint_path2 = os.path.abspath(os.path.join(config.out_dir2,config.checkpoint_dir))
if not os.path.isdir(os.path.abspath(config.checkpoint_path2)): os.makedirs(os.path.abspath(config.checkpoint_path2))

with tf.Graph().as_default() as c34fi_graph:
    session_conf3 = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    session_conf3.gpu_options.allow_growth = True
    sess3 = tf.Session(graph=c34fi_graph, config=session_conf3)
    with sess3.as_default():
        arnnC34_FI=AdversarialNetworkF(config, session=sess3)

        # Define Training procedure
        global_step3 = tf.Variable(0, name="global_step", trainable=False)
        optimizer3 = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars3 = optimizer3.compute_gradients(arnnC34_FI.loss)
        train_op3 = optimizer3.apply_gradients(grads_and_vars3, global_step=global_step3)
        sess3.run(tf.initialize_all_variables())
        
        saver3 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='labelpredictor_net'))
        # Load saved model to continue training or initialize all variables
        best_dir3 = os.path.join(config.checkpoint_path, "best_model_outputlayer")
        saver3.restore(sess3, best_dir3)
        print("Model loaded")

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
            grads_and_vars2 = optimizer.compute_gradients(net.loss)
            train_op2 = optimizer2.apply_gradients(grads_and_vars2, global_step=global_step2)
            
        #TODO vanaf hier!!!!

        print("Writing to {}\n".format(config.out_dir2))

        saver2 = tf.compat.v1.train.Saver(tf.all_variables(), max_to_keep=1)


        # initialize all variables
        print("Model initialized")
        #sess.run(tf.initialize_all_variables())
        best_dir2 = os.path.join(config.checkpoint_path2, 'best_model_acc')#"best_model_acc")
        saver.restore(sess2, best_dir2)
        print("Model loaded")


        def dev_step(x_batch, y_batch):
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
            }
            output_loss, total_loss, yhat, hiddenlayer = sess2.run(
                   [net.output_loss, net.loss, net.outputlayer, net.hiddenlayer], feed_dict)
            return np.sum(output_loss), total_loss, yhat,hiddenlayer

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

        def evaluate(gen):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss =0
            total_loss = 0
            yhat = np.zeros([128, len(gen.datalist)])
            hidden=np.zeros([40, len(gen.datalist)])
            num_batch_per_epoch = len(gen)
            test_step = 0
            yygt=np.zeros(len(gen.datalist))
            yyhatc34=np.zeros(len(gen.datalist))
            yyhateog=np.zeros(len(gen.datalist))
            yyhateogmapped =np.zeros(len(gen.datalist))
            ygt = np.zeros([128, len(gen.datalist)])
            x = np.zeros([128, len(gen.datalist)])
            while test_step < num_batch_per_epoch:
                (x_batch, y_batch) = gen[test_step]
                x_c=x_batch[:,0,:,:,0:3]
                x_eog=x_batch[:,0,:,:,1:4]
                y_batch=y_batch[:,0]
                output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnEOG.evalfeatures_adversarialnets( x_eog, y_batch)
#                    x_batch=np.moveaxis(x_batch,2,-1)                
#                    x_batch=np.moveaxis(x_batch,2,3)
                output_loss_, total_loss_, yhat_, hidden_ = dev_step(x_batch_eog, x_batch_c)
                output_loss1, total_loss1, yhat2, score1, temp = arnnC34_FI.evalfeatures_adversarialnets( yhat_, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                yhat[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(yhat_)
                ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_c)
                yygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                yyhatc34[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat0
                yyhateog[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat1
                yyhateogmapped[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2
                x[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_eog)
                hidden[:,(test_step)*config.batch_size : (test_step+1)*config.batch_size]=np.transpose(hidden_)
                test_step += 1
            if len(gen.datalist) - test_step*config.batch_size==1:
                yhat=yhat[:,0:-1]
                ygt=ygt[:,0:-1]
                x=x[:,0:-1]
                yygt=yygt[0:-1]
                yyhatc34=yyhatc34[:-1]
                yyhateog=yyhateog[:-1]
                yyhateogmapped=yyhateogmapped[:-1]
                hidden=hidden[:,0:-1]
            elif len(gen.datalist) > test_step*config.batch_size:
                (x_batch, y_batch) = gen[test_step]
                x_c=x_batch[:,0,:,:,0:3]
                x_eog=x_batch[:,0,:,:,1:4]
                y_batch=y_batch[:,0]
#                with sess.as_default():
                output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                output_loss1, total_loss1, yhat1, score1, x_batch_eog = arnnEOG.evalfeatures_adversarialnets( x_eog, y_batch)
#                with sess1.as_default():
                output_loss_, total_loss_, yhat_, hidden_ = dev_step(x_batch_eog, x_batch_c)
                output_loss1, total_loss1, yhat2, score1, temp = arnnC34_FI.evalfeatures_adversarialnets( yhat_, y_batch)
                yhat[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(yhat_)
                ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_c)
                x[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_eog)
                yygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
                yyhatc34[(test_step)*config.batch_size : len(gen.datalist)] = yhat0
                yyhateog[(test_step)*config.batch_size : len(gen.datalist)] = yhat1
                yyhateogmapped[(test_step)*config.batch_size : len(gen.datalist)] = yhat2
                hidden[:,(test_step)*config.batch_size : len(gen.datalist)]=np.transpose(hidden_)
                output_loss += output_loss_
                total_loss += total_loss_
            acc = accuracy_score(yygt, yyhatc34)
            print(acc)
            acc1 = accuracy_score(yygt, yyhateog)
            print(acc1)
            acc1 = accuracy_score(yygt, yyhateogmapped)
            print(acc1)
            return yhat, ygt, x, yygt, yyhatc34, yyhateog, yyhateogmapped,hidden

        print('Test')
        featf, feat, x , ygt, yyhatc34, yyhateog, yyhateogmapped,hidden = evaluate(gen=test_generatorC34)
        print('Train')
        featf2, feat2, x2, ygt2, yyhatc342, yyhateog2, yyhateogmapped2,hidden2 = evaluate(gen=train_generatorC34)
        print('Eval')
        featf3, feat3, x3, ygt3, yyhatc343, yyhateog3, yyhateogmapped3,hidden3 = evaluate(gen=eval_generatorC34)
        x_eog= np.transpose(np.concatenate([x, x2, x3],1))
        feat_c= np.transpose(np.concatenate([feat, feat2, feat3],1))
        feat_cf = np.transpose(np.concatenate([featf,featf2,featf3],1))
        hidden_c=np.transpose(np.concatenate([hidden, hidden2, hidden3],1))
        ygt_c= np.concatenate([ygt, ygt2, ygt3])+1
        yyhatc34_c= np.concatenate([yyhatc34, yyhatc342, yyhatc343])+1
        yyhateog_c= np.concatenate([yyhateog, yyhateog2, yyhateog3])+1
        yyhateogmapped_c = np.concatenate([yyhateogmapped, yyhateogmapped2, yyhateogmapped3])+1
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
        transh= reducer.fit(hidden_c)
        embeddingh= transh.transform(hidden_c)
        
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


