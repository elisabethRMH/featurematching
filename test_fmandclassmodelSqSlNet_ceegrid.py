'''
Testing the feature matching model with SeqSleepNet on the cEEGrid dataset.
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1" #force not using GPU!
import numpy as np
import tensorflow as tf
import math

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat
import copy
import umap
import seaborn as sns

#from arnn_sleep_sup import ARNN_Sleep
from FMandClassModel_SeqSlNet import FMandClass_ModelSeqSlNet
from fmandclassmodel_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap
import matplotlib.pyplot as plt

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy
datapath='/esat/stadiusdata/sensitive/cEEGrid/'

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

normalize=True
number_patients=5
#VERSION WITH PATIENT GROUPS
dim=20*int(10/number_patients)
acc_matrix=np.zeros((dim,4))
kap_matrix=np.zeros((dim,4))
acc_matrixC=np.zeros((dim,4))
kap_matrixC=np.zeros((dim,4))
ind=0
#VERSION WITH PATIENT GROUPS
for fold in range(12):
    for pat_group in range(int(10/number_patients)):
                
                
    
        eeg_train_data= "/esat/asterie1/scratch/ehereman/data_processing_SeqSlNet/tf_data3/seqsleepnet_eeg/train_list_total.txt".format(fold+1)
        eeg_retrain_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/train_list_n{:d}.txt".format(fold+1)
        eeg_eval_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/eval_list_n{:d}.txt".format(fold+1)
        eeg_test_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/test_list_n{:d}.txt".format(fold+1)
        eog_retrain_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/train_list_n{:d}.txt".format(fold+1)
        eog_eval_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/eval_list_n{:d}.txt".format(fold+1)
        eog_test_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/test_list_n{:d}.txt".format(fold+1)
        replacepaths= ['/esat/asterie1/scratch/ehereman/cEEGGrid/cEEGGrid', datapath]
        # My Parameters 

        config= Config()
        
        order=  [0,2,3,4,1] #REM is in second place and is placed fifth. 
        
        list1= [eeg_eval_data, eog_eval_data]
        eval_generator= SubGenFromFileHuy(filelist_lst=list1,shuffle=False, batch_size=config.batch_size,  sequence_size=config.epoch_seq_len, normalize_per_subject=True, replacepaths=replacepaths)
        eval_generator.y=eval_generator.y[:,order]
    
        list1= [eeg_test_data, eog_test_data]
        test_generator=SubGenFromFileHuy(filelist_lst=list1,shuffle=False, batch_size=config.batch_size, sequence_size=config.epoch_seq_len, normalize_per_subject=True, replacepaths=replacepaths)
        test_generator.y=test_generator.y[:,order]
        
        
            
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
        
#        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_subjnorm_totalmass2/n{:d}'.format( fold), "Point to output directory")
        dir2 = '/esat/asterie1/scratch/ehereman/results_featuremapping/ceegrid/seqslnet_fmandclasstraining_eogtoc34_samenetwork_subjnorm_eog{:d}pat'.format(number_patients)
        dir2 = '/esat/asterie1/scratch/ehereman/results_featuremapping/ceegrid/seqslnet_fmandclasstraining_eogtoc34_diffnetwork_subjnorm_eog{:d}pat'.format(number_patients)
        # dir2= '/esat/asterie1/scratch/ehereman/results_featuremapping/ceegrid/mmd/seqslnet_fmandclasstraining_eogtoc34_diffnetwork_subjnorm_eog{:d}pat'.format(number_patients)
        tf.app.flags.DEFINE_string("out_dir", dir2+'/n{:d}/group{:d}'.format( fold, pat_group), "Point to output directory")

        tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
        
        tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
        
        tf.app.flags.DEFINE_integer("seq_len", 10, "Sequence length (default: 32)")
        
        tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
        
        tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
        tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 20)")
        
        
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

        
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.training_epoch = int(20) 
        config.same_network=False
        config.feature_extractor=True
        config.learning_rate=1e-4
        config.mse_loss=True
        config.mse_weight=1.0
        config.mult_channel=False
        config.withEOGclass=True
        config.mmd_loss= False
        config.mmd_weight=1.0        
        config.diffattn=False
        
#        train_batches_per_epoch = np.floor(len(retrain_generator)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
                
        
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
                arnn=FMandClass_ModelSeqSlNet(config)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

                best_dir = os.path.join(checkpoint_path, "best_model_acc")
                saver.restore(sess, best_dir)
                print("Model loaded")
        
        
        
                def evalfeatures_eog(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                        arnn.net2active:np.ones(len(x_batch)),
                        arnn.net1active:np.ones(len(x_batch)),
                        arnn.input_x: x_batch,
                        arnn.input_y: y_batch,
                        arnn.dropout_keep_prob_rnn: 1.0,
                        arnn.frame_seq_len: frame_seq_len,
                        arnn.epoch_seq_len: epoch_seq_len,
                        arnn.training:False
                    }
                    output_loss, mse_loss, total_loss, yhat, yhateog, score, scoreeog, features1, features2 = sess.run(
                           [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.predictions, arnn.predictions_eog, arnn.scores, arnn.scores_eog, arnn.features1, arnn.features2], feed_dict)
                    return output_loss, mse_loss,total_loss, yhat, yhateog, score, scoreeog, features1, features2
            
                def evaluate(gen):
                    # Validate the model on the entire evaluation test set after each epoch
        
                    output_loss =0
                    total_loss = 0
                    mse_loss=0
                    yhat = np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    yhateog =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    score = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    scoreeog = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    featC = np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    featEOG= np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    while test_step < num_batch_per_epoch-1:
                        (x_batch,y_batch)=gen[test_step]
        
                        output_loss_, mse_loss_, total_loss_, yhat_, yhat2, score_, scoreeog_, features1_,features2_ = evalfeatures_eog(x_batch, y_batch)
                        output_loss += output_loss_
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                        
                        featC[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features1_)
                        featEOG[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features2_)

                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            yhateog[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2[n]
                            score[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = score_[n]
                            scoreeog[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = scoreeog_[n]

                        test_step += 1
                            
                    if len(gen.datalist) > test_step*config.batch_size:
                        (x_batch, y_batch) = gen.get_rest_batch(test_step)
        
                        output_loss_, mse_loss_, total_loss_, yhat_, yhat2, score_, scoreeog_, features1_,features2_ = evalfeatures_eog(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                            yhateog[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat2[n]
                            score[n, (test_step)*config.batch_size : len(gen.datalist),:] = score_[n]
                            scoreeog[n, (test_step)*config.batch_size : len(gen.datalist),:] = scoreeog_[n]

                        featC[:,:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features1_)
                        featEOG[:,:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features2_)
        
                        output_loss += output_loss_
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                    yhat = yhat + 1
                    ygt= ygt+1
                    yhateog+=1
                    acc = accuracy_score(ygt.flatten(), yhat.flatten())
                    print(acc)
                    acc1 = accuracy_score(ygt.flatten(), yhateog.flatten())
                    print(acc1)
                    return featC, featEOG, ygt, yhat, yhateog,score, scoreeog


                print('Test')
                feat_c1, feat_eog1, ygt1, yyhatc341, yyhateog1, scorec341, scoreeog1 = evaluate(gen=test_generator)
                print('Train')
#                feat_c2, feat_eog2, ygt2, yyhatc342, yyhateog2, scorec342, scoreeog2 = evaluate(gen=train_generator)
                print('Eval')
                feat_c3, feat_eog3, ygt3, yyhatc343, yyhateog3, scorec343, scoreeog3  = evaluate(gen=eval_generator)
                print('retrain')
#                feat_c4, feat_eog4, ygt4, yyhatc344, yyhateog4, scorec344, scoreeog4  = evaluate(gen=retrain_generator)
        
                savemat(os.path.join(out_path, "test_retEOG.mat"), dict(yhat = yyhateog1, acc = accuracy_score(ygt1.flatten(), yyhateog1.flatten()),kap=kap(ygt1.flatten(), yyhateog1.flatten()),
                                                                     ygt=ygt1, subjects=test_generator.subjects_datalist, score=scoreeog1, scorec34=scorec341)  )              
                savemat(os.path.join(out_path, "eval_retEOG.mat"), dict(yhat = yyhateog3, acc = accuracy_score(ygt3.flatten(), yyhateog3.flatten()),kap=kap(ygt3.flatten(), yyhateog3.flatten()),
                                                                     ygt=ygt3, subjects=eval_generator.subjects_datalist, score=scoreeog3, scorec34=scorec343)  )               

                feat_c= np.transpose(np.concatenate([feat_c1[:,0,:],  feat_c3[:,0,:]],1))
                feat_eog = np.transpose(np.concatenate([feat_eog1[:,0,:],feat_eog3[:,0,:]],1))
                ygt_c= np.concatenate([ygt1[0,:], ygt3[0,:]])
                yyhatc34_c= np.concatenate([yyhatc341[0,:], yyhatc343[0,:]])
                yyhateog_c= np.concatenate([yyhateog1[0,:], yyhateog3[0,:]])
                dff_feat=np.mean(np.abs(feat_c-feat_eog))
                meansqdff=np.mean(np.power(feat_c-feat_eog,2))




reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)

trans= reducer.fit(feat_c)
embeddingc=trans.transform(feat_c)
embeddingeog2=trans.transform(feat_eog)

colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_c]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embeddingeog2[:,0], embeddingeog2[:,1],color=colors, s=.1)
