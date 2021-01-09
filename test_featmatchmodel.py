'''
Testing of a feature matching model: an arnn model that learns to match features of a
modality as best as possible to corresponding features of another modality, calculated by a 
different model (that is previously trained).
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf
import umap

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from arnn_sleep_sup import ARNN_Sleep
from FeatMatchModel import FeatMatch_Model
sys.path.insert(1,'/users/sista/ehereman/Documents/code/adversarial_DA/')
from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config
from adversarialnetwork_featureinput import AdversarialNetworkF

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap

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

nb_patients= 2
#fold=0
#print('Fold: ', fold)
train_files=files_folds['train_sub']#[fold][0][0]
eval_files=files_folds['eval_sub']#[fold][0][0]
test_files=files_folds['test_sub']#[fold][0][0]

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

tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")

tf.app.flags.DEFINE_integer("seq_len", 32, "Sequence length (default: 32)")

tf.app.flags.DEFINE_integer("nfilter", 20, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 32, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer('D',100,'Number of features') #new flag!


config = Config()

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
config.training_epoch=max(60,int(60*10/nb_patients))#25
#    config.training_epoch = int(60) #/6 if using load_random_tuple
config.domainclassifier=False
config.domainclassifierstage2=False
config.add_classifierinput=False
config.out_path='' #This is just to be compatible with adversarialnetwork.py (not actually being used)
config.checkpoint_path= ''
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



di2=np.load('/users/sista/ehereman/Documents/code/feature_mapping/patient_groups.npy',allow_pickle=True)
di2=di2.item()
acc_matrix=np.zeros((len(di2[nb_patients]),3))
kap_matrix=np.zeros((len(di2[nb_patients]),3))

for pat_group in range(1,len(di2[nb_patients])):
    pat=di2[nb_patients][pat_group]
    
    savedir='/esat/asterie1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_samenetwork2_eog{:d}pat'.format(nb_patients)
    config.out_dir=os.path.join(savedir,'group{:d}'.format( pat_group))
    config.out_dir1 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum'.format(nb_patients)
    
    FLAGS = tf.app.flags.FLAGS
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()): # python3
        print("{}={}".format(attr.upper(), value))
    print("")
    
    # Data Preparatopn
    # ==================================================
    
    # path where some output are stored
    out_path = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
    out_path1= os.path.join(config.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
    # path where checkpoint models are stored
    checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
    if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))
    
    checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
    if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
    
    
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
            
    with tf.Graph().as_default() as c34fi_graph:
        session_conf3 = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
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
            best_dir3 = os.path.join(checkpoint_path, "best_model_acc")
            saver3.restore(sess3, best_dir3)
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
    
            out_dir = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
            print("Writing to {}\n".format(out_dir))
    
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    
            # initialize all variables
            print("Model initialized")
    #            sess.run(tf.initialize_all_variables())
            best_dir = os.path.join(checkpoint_path, "best_model_acc")
            saver.restore(sess, best_dir)
            print("Model loaded")
    
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
                mse_loss, total_loss,featEOG, featC= sess.run(
                       [ arnn.mse_loss, arnn.loss, arnn.features2, arnn.features1], feed_dict)
                return mse_loss, total_loss, featEOG, featC
    
            def evaluate(gen):
                # Validate the model on the entire evaluation test set after each epoch
    
                num_batch_per_epoch = len(gen)
                test_step = 0
                ygt = np.zeros(len(gen.datalist))
                total_loss = 0
                mse_loss=0
                yhatc = np.zeros(len(gen.datalist))
                yhateog =np.zeros(len(gen.datalist))
                featC = np.zeros([128, len(gen.datalist)])
                featEOG= np.zeros([128, len(gen.datalist)])
                while test_step < num_batch_per_epoch:
                    #((x_batch, y_batch),_) = gen[test_step]
                    (x_batch,y_batch)=gen[test_step]
    #                    x_batch=x_batch[:,0]
                    x_c=x_batch[:,0,:,:,0:3]
                    x_eog=x_batch[:,0,:,:,1:4]
                    y_batch=y_batch[:,0]
                    if not config.same_network:
                        output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                        mse_loss_, total_loss_, x_batch_eog, x_batch_c2 = dev_step(x_eog, y_batch,x_batch_c)
#                        assert(x_batch_c==x_batch_c2)
                        
                    else:
                        mse_loss_, total_loss_, x_batch_eog, x_batch_c = dev_step(x_eog, y_batch,x_c)
                        output_loss0, total_loss0, yhat0, scor0, temp = arnnC34_FI.evalfeatures_adversarialnets( x_batch_c, y_batch)
                    output_loss1, total_loss1, yhat2, score1, temp = arnnC34_FI.evalfeatures_adversarialnets( x_batch_eog, y_batch)
                    
                    total_loss += total_loss_
                    mse_loss+= mse_loss_
                    featC[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_c)
                    ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                    yhatc[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat0
                    yhateog[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2
                    featEOG[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_eog)
                    
    #                    yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
    #                    ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                    test_step += 1
                if len(gen.datalist) - test_step*config.batch_size==1:
                    yhateog=yhateog[0:-1]
                    ygt=ygt[0:-1]
                    yhatc=yhatc[0:-1] 
                    featC=featC[:,0:-1]
                    featEOG=featEOG[:,0:-1]
                elif len(gen.datalist) > test_step*config.batch_size:
                    # if using load_random_tuple
                    #((x_batch, y_batch),_) = gen[test_step]
                    (x_batch,y_batch)=gen[test_step]
    #                    x_batch=x_batch[:,0]
                    x_c=x_batch[:,0,:,:,0:3]
                    x_eog=x_batch[:,0,:,:,1:4]
                    y_batch=y_batch[:,0]
                    if not config.same_network:
                        output_loss0, total_loss0, yhat0, scor0, x_batch_c = arnnC34.evalfeatures_adversarialnets(x_c,y_batch)
                        mse_loss_, total_loss_, x_batch_eog, x_batch_c2 = dev_step(x_eog, y_batch,x_batch_c)
                        assert(x_batch_c==x_batch_c2)
                    else:
                        mse_loss_, total_loss_, x_batch_eog, x_batch_c = dev_step(x_eog, y_batch,x_c)
                        output_loss0, total_loss0, yhat0, scor0, temp = arnnC34_FI.evalfeatures_adversarialnets( x_batch_c, y_batch)
                    output_loss1, total_loss1, yhat2, score1, temp = arnnC34_FI.evalfeatures_adversarialnets( x_batch_eog, y_batch)
    #                    ygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
    #                    yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
                    total_loss += total_loss_
                    mse_loss+= mse_loss_
                    featC[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_c)
                    ygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
                    yhatc[(test_step)*config.batch_size : len(gen.datalist)] = yhat0
                    yhateog[(test_step)*config.batch_size : len(gen.datalist)] = yhat2
                    featEOG[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_eog)
    
                yhatc = yhatc + 1
                ygt= ygt+1
                yhateog=yhateog+1
                acc = accuracy_score(ygt, yhatc)
                print(acc)
                acc1 = accuracy_score(ygt, yhateog)
                print(acc1)
                return featEOG, featC, ygt, yhatc,  yhateog
    
            print('Test')
            featf, feat, ygt, yyhatc34,  yyhateogmapped = evaluate(gen=test_generator)
            savemat(os.path.join(out_path, "test_retEOG.mat"), dict(yhat = yyhateogmapped, acc = accuracy_score(ygt, yyhateogmapped),kap=kap(ygt, yyhateogmapped),
                                                                 ygt=ygt))                
            print('Train')
            featf2, feat2, ygt2, yyhatc342,  yyhateogmapped2 = evaluate(gen=train_generator)
            savemat(os.path.join(out_path, "train_retEOG.mat"), dict(yhat = yyhateogmapped2, acc = accuracy_score(ygt2, yyhateogmapped2),kap=kap(ygt2, yyhateogmapped2),
                                                                 ygt=ygt2))                
            print('Eval')
            featf3, feat3,  ygt3, yyhatc343,yyhateogmapped3 = evaluate(gen=eval_generator)
            savemat(os.path.join(out_path, "eval_retEOG.mat"), dict(yhat = yyhateogmapped3, acc = accuracy_score(ygt3, yyhateogmapped3),kap=kap(ygt3, yyhateogmapped3),
                                                                 ygt=ygt3))                
            acc_matrix[pat_group,:]=np.array((accuracy_score(ygt,yyhateogmapped),accuracy_score(ygt3,yyhateogmapped3),accuracy_score(ygt2,yyhateogmapped2)))
            kap_matrix[pat_group,:]=np.array((kap(ygt,yyhateogmapped),kap(ygt3,yyhateogmapped3),kap(ygt2,yyhateogmapped2)))
            feat_c= np.transpose(np.concatenate([feat, feat2, feat3],1))
            feat_cf = np.transpose(np.concatenate([featf,featf2,featf3],1))
            ygt_c= np.concatenate([ygt, ygt2, ygt3])
            yyhatc34_c= np.concatenate([yyhatc34, yyhatc342, yyhatc343])
            yyhateogmapped_c = np.concatenate([yyhateogmapped, yyhateogmapped2, yyhateogmapped3])
            dff_feat=np.mean(np.abs(feat_c-feat_cf))
            meansqdff=np.mean(np.power(feat_c-feat_cf,2))
            print(dff_feat)
            print(meansqdff)
    
            reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
            
            trans= reducer.fit(feat_c)
            embeddingc=trans.transform(feat_c)
            embeddingf2=trans.transform(feat_cf)
#            transf= reducer.fit(feat_cf)
#            embeddingf= transf.transform(feat_cf)
#            embeddingc2=transf.transform(feat_c)
#            
#            #embedding_t= reducer.fit_transform(feat_t)
#            
#    #        #colors_age=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_c]
#    #        colors_age=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_c]
#    #        #colors_aget=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_t]
#    #        colors_aget=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_t]
#            
            colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_c]
#    #        colorst=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_t]
#            #sns.palplot(sns.color_palette('hls',7))
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.scatter(embeddingf2[:,0], embeddingf2[:,1],color=colors, s=.1)
#    
#            fig=plt.figure()
#            ax=fig.add_subplot(111)        
#            ax.scatter(embeddingf[:,0], embeddingf[:,1],color=colors, s=.1)
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            ax.scatter(embeddingc2[:,0], embeddingc2[:,1],color=colors, s=.1)
#    

print(np.mean(acc_matrix,0))
print(np.median(acc_matrix,0))
print(np.mean(kap_matrix,0))
np.save('/esat/asterie1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_diffnetwork32_eog{:d}pat/acc_matrix.npy'.format(nb_patients),acc_matrix )

fig1, ax1 = plt.subplots()
ax1.set_title('Accuracies for feature matching method \n trained on groups of {:d} patients \n (same feature extractor network for C34 and EOG)'.format(nb_patients))
ax1.set_ylabel('Sleep staging accuracy')
ax1.boxplot(acc_matrix,labels=['Test set' , 'Validation set','Training set'])
ax1.scatter(np.ones(acc_matrix.shape)*np.array([1,2,3]),acc_matrix)        
