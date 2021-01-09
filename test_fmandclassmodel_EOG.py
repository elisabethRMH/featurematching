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
import tensorflow as tf
import umap
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import copy

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from arnn_sleep_sup import ARNN_Sleep
from FMandClassModel import FMandClass_Model
from fmandclassmodel_config import Config

sys.path.insert(1,'/users/sista/ehereman/Documents/code/adversarial_DA/')
from adversarialnetwork_featureinput import AdversarialNetworkF

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
#from subgenfromfile_loadrandomtuples import SubGenFromFile
from subgenfromfile_epochsave import SubGenFromFile

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch4'; # no overlap
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap


nb_patients=10

di2=np.load('patient_groups.npy',allow_pickle=True)
di2=di2.item()
acc_matrix=np.zeros((len(di2[nb_patients]),3))
kap_matrix=np.zeros((len(di2[nb_patients]),3))

for pat_group in range(len(di2[nb_patients])):
    
    pat=di2[nb_patients][pat_group]
    test_files=files_folds['test_sub']#[fold][0][0]
    eval_files=files_folds['eval_sub']#[fold][0][0]
    train_files=files_folds['train_sub']
    train_files_forEOGnet=files_folds['train_sub'][:,pat]#[fold][0][0]



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
    tf.app.flags.DEFINE_string("out_dir", '/volume1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork6_eog{:d}pat/group{:d}'.format(nb_patients, pat_group), "Point to output directory")
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
    best_acc = 0.0
    best_kappa = 0.0
    min_loss = float("inf")
    # Training
    # ==================================================
    with tf.Graph().as_default() as c34fi_graph:
        session_conf1 = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        session_conf1.gpu_options.allow_growth = True
        sess1 = tf.Session(graph=c34fi_graph, config=session_conf1)
        with sess1.as_default():
            config1=copy.copy(config)
            config1.feature_extractor=False
            arnnC34_FI=FMandClass_Model(config1)
    
            # Define Training procedure
            global_step1 = tf.Variable(0, name="global_step", trainable=False)
            optimizer1 = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars1 = optimizer1.compute_gradients(arnnC34_FI.loss)
            train_op1 = optimizer1.apply_gradients(grads_and_vars1, global_step=global_step1)
            sess1.run(tf.initialize_all_variables())
            
            saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='labelpredictor_net'))
            # Load saved model to continue training or initialize all variables
            best_dir1 = os.path.join(checkpoint_path, "best_model_outputlayer_acc")
            saver1.restore(sess1, best_dir1)
            print("Model loaded")
    
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
    
            saver = tf.train.Saver(tf.all_variables())
            # Load saved model to continue training or initialize all variables
            best_dir = os.path.join(checkpoint_path, "best_model_acc")
            saver.restore(sess, best_dir)
            print("Model loaded")
    
    #        def train_step(x_batch, y_batch): #not adapted
    #            """
    #            A single training step
    #            """
    #            frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
    #            feed_dict = {
    #              arnn.input_x: x_batch,
    #              arnn.input_y: y_batch,
    #              arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
    #              arnn.frame_seq_len: frame_seq_len
    #            }
    #            _, step, output_loss,mse_loss, total_loss, accuracy = sess.run(
    #               [train_op, global_step, arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.accuracy],
    #               feed_dict)
    #            return step, output_loss, mse_loss, total_loss, accuracy
    #
    #        def dev_step(x_batch, y_batch):
    #            frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
    #            feed_dict = {
    #                arnn.input_x: x_batch,
    #                arnn.input_y: y_batch,
    #                arnn.dropout_keep_prob_rnn: 1.0,
    #                arnn.frame_seq_len: frame_seq_len
    #            }
    #            output_loss, mse_loss, total_loss, yhat = sess.run(
    #                   [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.prediction], feed_dict)
    #            return output_loss, mse_loss, total_loss, yhat
    
            def evalfeatures_eog(x_batch, y_batch):
                frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
                feed_dict = {
                    arnn.net2active:np.ones(len(x_batch)),
                    arnn.input_x: x_batch,
                    arnn.input_y: y_batch,
                    arnn.dropout_keep_prob_rnn: 1.0,
                    arnn.frame_seq_len: frame_seq_len
                }
                output_loss, mse_loss, total_loss, yhat, score, features1, features2 = sess.run(
                       [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.prediction, arnn.score, arnn.features1, arnn.features2], feed_dict)
                return output_loss, mse_loss,total_loss, yhat, score, features1, features2
            def evalscores_eog(features_eog, features_c34,y_batch):
                feed_dict = {
                    arnnC34_FI.net2active:np.ones(len(y_batch)),
                    arnnC34_FI.input_y: y_batch,
                    arnnC34_FI.dropout_keep_prob_rnn: 1.0,
                    arnnC34_FI.features1:features_eog,
                    arnnC34_FI.features2:features_c34
                }
                output_loss, mse_loss, total_loss, yhat, score= sess1.run(
                       [arnnC34_FI.output_loss, arnnC34_FI.mse_loss, arnnC34_FI.loss, arnnC34_FI.prediction, arnnC34_FI.score], feed_dict)
                return yhat
            
            def evaluate(gen):
                # Validate the model on the entire evaluation test set after each epoch
    
                output_loss =0
                total_loss = 0
                mse_loss=0
                yhat = np.zeros(len(gen.datalist))
                yyhateogmapped =np.zeros(len(gen.datalist))
                num_batch_per_epoch = len(gen)
                test_step = 0
                ygt = np.zeros(len(gen.datalist))
                featC = np.zeros([128, len(gen.datalist)])
                featEOG= np.zeros([128, len(gen.datalist)])
                while test_step < num_batch_per_epoch:
                    #((x_batch, y_batch),_) = gen[test_step]
                    (x_batch,y_batch)=gen[test_step]
                    x_batch=x_batch[:,0]
                    y_batch=y_batch[:,0]
    
                    output_loss_, mse_loss_, total_loss_, yhat_, score_, features1_,features2_ = evalfeatures_eog(x_batch, y_batch)
                    yhat2 = evalscores_eog( features2_,features1_, y_batch)
                    output_loss += output_loss_
                    total_loss += total_loss_
                    mse_loss+= mse_loss_
                    
                    featC[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features1_)
                    featEOG[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features2_)
                    yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
                    ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=1))
                    yyhateogmapped[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2
                    test_step += 1
                if len(gen.datalist) - test_step*config.batch_size==1:
                    yhat=yhat[0:-1]
                    ygt=ygt[0:-1]
                    yyhateogmapped=yyhateogmapped[:-1]
                    featC=featC[:,0:-1]
                    featEOG= featEOG[:,0:-1]
                        
                elif len(gen.datalist) > test_step*config.batch_size:
                    # if using load_random_tuple
                    #((x_batch, y_batch),_) = gen[test_step]
                    (x_batch,y_batch)=gen[test_step]
                    x_batch=x_batch[:,0]
                    y_batch=y_batch[:,0]
    
                    output_loss_, mse_loss_, total_loss_, yhat_, score_, features1_,features2_ = evalfeatures_eog(x_batch, y_batch)
                    yhat2 = evalscores_eog( features2_,features1_, y_batch)
                    ygt[(test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=1))
                    yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
                    yyhateogmapped[(test_step)*config.batch_size : len(gen.datalist)] = yhat2
                    featC[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features1_)
                    featEOG[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features2_)
    
                    output_loss += output_loss_
                    total_loss += total_loss_
                    mse_loss+= mse_loss_
                yhat = yhat + 1
                ygt= ygt+1
                yyhateogmapped+=1
                acc = accuracy_score(ygt, yhat)
                print(acc)
                acc1 = accuracy_score(ygt, yyhateogmapped)
                print(acc1)
                return ygt, yhat, yyhateogmapped, featC, featEOG
    
            ygt, yyhatc34, yyhateogmapped,featC, featEOG = evaluate(gen=test_generator)
            ygt2, yyhatc342, yyhateogmapped2,featC2, featEOG2 = evaluate(gen=train_generator)
            ygt3, yyhatc343, yyhateogmapped3,featC3, featEOG3 = evaluate(gen=eval_generator)
            feat_c= np.transpose(np.concatenate([featC, featC2, featC3],1))
            feat_cf = np.transpose(np.concatenate([featEOG,featEOG2,featEOG3],1))
            ygt_c= np.concatenate([ygt, ygt2, ygt3])
            yyhatc34_c= np.concatenate([yyhatc34, yyhatc342, yyhatc343])
            yyhateogmapped_c = np.concatenate([yyhateogmapped, yyhateogmapped2, yyhateogmapped3])
            dff_feat=np.mean(np.abs(feat_c-feat_cf))
            meansqdff=np.mean(np.power(feat_c-feat_cf,2))
            print(dff_feat)
            print(meansqdff)
            acc_matrix[pat_group,:]=np.array((accuracy_score(ygt,yyhateogmapped),accuracy_score(ygt3,yyhateogmapped3),accuracy_score(ygt2,yyhateogmapped2)))
            kap_matrix[pat_group,:]=np.array((kap(ygt,yyhateogmapped),kap(ygt3,yyhateogmapped3),kap(ygt2,yyhateogmapped2)))
            
            print('Acc C34', 'test', np.sum(yyhatc34==ygt)/len(ygt), 'eval', np.sum(yyhatc343==ygt3)/len(ygt3))
            
#            reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
#            trans= reducer.fit(feat_c)
#            embeddingc=trans.transform(feat_c)
#            embeddingf2=trans.transform(feat_cf)
#            transf= reducer.fit(feat_cf)
#            embeddingf= transf.transform(feat_cf)
#            embeddingc2=transf.transform(feat_c)
    
            
            #embedding_t= reducer.fit_transform(feat_t)
            
    #        #colors_age=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_c]
    #        colors_age=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_c]
    #        #colors_aget=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_t]
    #        colors_aget=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_t]
            
#            colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_c]
#    #        colorst=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_t]
#            #sns.palplot(sns.color_palette('hls',7))
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            ax.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            ax.scatter(embeddingf2[:,0], embeddingf2[:,1],color=colors, s=.1)
#    
#            fig=plt.figure()
#            ax=fig.add_subplot(111)        
#            ax.scatter(embeddingf[:,0], embeddingf[:,1],color=colors, s=.1)
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            ax.scatter(embeddingc2[:,0], embeddingc2[:,1],color=colors, s=.1)
    
print(np.mean(acc_matrix,0))
print(np.median(acc_matrix,0))
print(np.mean(kap_matrix,0))
np.save('/esat/asterie1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork6_eog{:d}pat/acc_matrix.npy'.format(nb_patients),acc_matrix )
np.save('/esat/asterie1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork6_eog{:d}pat/kap_matrix.npy'.format(nb_patients),kap_matrix )
