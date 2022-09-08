'''
Feature matching with SeqSleepNet on cEEGrid
Components
- source feature extractor
- target feature extractor
- source classifier
- target classifier
Loss: MSE loss + classification loss source + class loss target
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf
import math

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat

from FMandClassModel_SeqSlNet import FMandClass_ModelSeqSlNet
from fmandclassmodel_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy
datapath='/esat/stadiusdata/sensitive/cEEGrid/'
sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

normalize=True
number_patients=5
#VERSION WITH PATIENT GROUPS
for fold in range(12):
    for pat_group in range(int(10/number_patients)):
                
        fileidx=np.arange(pat_group* number_patients,(pat_group+1)*number_patients)
    
        source_train_data= "/esat/asterie1/scratch/ehereman/data_processing_SeqSlNet/tf_data3/seqsleepnet_eeg/train_list_total.txt".format(fold+1)
        source_retrain_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/train_list_n{:d}.txt".format(fold+1)
        source_eval_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/eval_list_n{:d}.txt".format(fold+1)
        source_test_data= datapath+"_PSG_mat/tf_data/seqsleepnet_eeg/test_list_n{:d}.txt".format(fold+1)
        target_retrain_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/train_list_n{:d}.txt".format(fold+1)
        target_eval_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/eval_list_n{:d}.txt".format(fold+1)
        target_test_data= datapath+"_cEEGrid_mat/tf_data/seqsleepnet_eeg/test_list_n{:d}.txt".format(fold+1)
        replacepaths= ['/esat/asterie1/scratch/ehereman/cEEGGrid/cEEGGrid', datapath]
        # My Parameters 

        config= Config()
        
        order=  [0,2,3,4,1] #REM is in second place and is placed fifth. 
        
        list1= [source_retrain_data, target_retrain_data]
        batch_size=8
        retrain_generator= SubGenFromFileHuy(filelist_lst=list1, fileidx=fileidx,shuffle=True, batch_size=batch_size, sequence_size=config.epoch_seq_len, normalize_per_subject=True, replacepaths=replacepaths)
        retrain_generator.y=retrain_generator.y[:,order]

        list1= [source_train_data]
        train_generator= SubGenFromFileHuy(filelist_lst=list1,shuffle=True, batch_size=config.batch_size,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, replacepaths=replacepaths)
        train_generator.batch_size=min(int(np.floor(len(train_generator.datalist)/len(retrain_generator))),200)
        
        list1= [source_eval_data, target_eval_data]
        eval_generator= SubGenFromFileHuy(filelist_lst=list1,shuffle=False, batch_size=config.batch_size,  sequence_size=config.epoch_seq_len, normalize_per_subject=True, replacepaths=replacepaths)
        eval_generator.y=eval_generator.y[:,order]
    
        list1= [source_test_data, target_test_data]
        test_generator=SubGenFromFileHuy(filelist_lst=list1,shuffle=False, batch_size=config.batch_size,  sequence_size=config.epoch_seq_len, normalize_per_subject=True, replacepaths=replacepaths)
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
        
        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_SeqSleepNet_tb/totalmass2/seqsleepnet_sleep_nfilter32_seq10_dropout0.75_nhidden64_att64_1chan_subjnorm/total', "Point to output directory")
        tf.app.flags.DEFINE_string("out_dir", '/esat/asterie1/scratch/ehereman/results_featuremapping/ceegrid/seqslnet_fmandclasstraining_targettosource_diffnetwork_subjnorm_target{:d}pat/n{:d}/group{:d}'.format(number_patients, fold, pat_group), "Point to output directory")
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

        out_path1= FLAGS.out_dir1 #os.path.join(FLAGS.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        # path where checkpoint models are stored
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
        config.training_epoch = int(20) 
        config.same_network=False
        config.feature_extractor=True
        config.learning_rate=1e-4
        config.mse_loss=True
        config.mse_weight=1.0
        config.mse_weight=min(config.mse_weight*(train_generator.batch_size+retrain_generator.batch_size)/(retrain_generator.batch_size),1500)
        config.mult_channel=False
        config.withtargetclass=True
        config.mmd_loss= False
        config.mmd_weight=1.0        
        config.mmd_weight=config.mmd_weight*(train_generator.batch_size+retrain_generator.batch_size)/(retrain_generator.batch_size)
        config.diffattn=False
        
        train_batches_per_epoch = np.floor(len(retrain_generator)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
        
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
                arnn=FMandClass_ModelSeqSlNet(config)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer/output-'),max_to_keep=1)
                sess.run(tf.initialize_all_variables())
                saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                var_list1 = {}
                for v1 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'output_layer/outputtarget'):
                    tmp = v1.name.replace('outputtarget','output')
                    tmp=tmp[:-2]
                    var_list1[tmp]=v1
                saver1=tf.train.Saver(var_list=var_list1)
                
                saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                var_list2= {}
                for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source'):
                    tmp=v2.name[16:-2]
                    var_list2[tmp]=v2
                saver2=tf.train.Saver(var_list=var_list2)
                saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                
                if not config.same_network:
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target'):
                        tmp=v2.name[16:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer'),max_to_keep=1)
                print("Model loaded")
        
                def train_step(x_batch, y_batch,net2bool): #not adapted
                    """
                    A single training step
                    """
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                      arnn.net2active:net2bool,
                      arnn.net1active: 1-net2bool,
                      arnn.input_x: x_batch,
                      arnn.input_y: y_batch,
                      arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                      arnn.frame_seq_len: frame_seq_len,
                      arnn.epoch_seq_len: epoch_seq_len,
                      arnn.training: True
                    }
                    _, step, output_loss,mse_loss, total_loss, accuracy = sess.run(
                       [train_op, global_step, arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.accuracy],
                       feed_dict)
                    return step, output_loss, mse_loss, total_loss, np.mean(accuracy)
        
                def dev_step(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                        arnn.net2active:np.ones(len(x_batch)),
                        arnn.net1active: np.ones(len(x_batch)),
                        arnn.input_x: x_batch,
                        arnn.input_y: y_batch,
                        arnn.dropout_keep_prob_rnn: 1.0,
                        arnn.frame_seq_len: frame_seq_len,
                        arnn.epoch_seq_len: epoch_seq_len,
                        arnn.training: False
                    }
                    output_loss, mse_loss, total_loss, yhat, yhattarget = sess.run(
                           [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.predictions, arnn.predictions_target], feed_dict)
                    return output_loss, mse_loss, total_loss, (yhat), yhattarget
        

                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
                    datalstlen=len(gen.datalist)
                    output_loss =0
                    total_loss = 0
                    mse_loss=0

                    yhat = np.zeros([config.epoch_seq_len, datalstlen])
                    yhattarget = np.zeros([config.epoch_seq_len, datalstlen])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt = np.zeros([config.epoch_seq_len, datalstlen])
                
                    while test_step < num_batch_per_epoch-1:

                        (x_batch,y_batch)=gen[test_step]

        
                        output_loss_, mse_loss_, total_loss_, yhat_, yhattarget_= dev_step(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhattarget_[n]
                        output_loss += output_loss_
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                        test_step += 1
                            
                    if len(gen.datalist) > test_step*config.batch_size:
                        (x_batch,y_batch)=gen.get_rest_batch(test_step)

        
                        output_loss_, mse_loss_, total_loss_, yhat_, yhattarget_= dev_step(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : len(gen.datalist)] = yhattarget_[n]
                        output_loss += output_loss_
                        total_loss += total_loss_
                        mse_loss+= mse_loss_
                    yhat = yhat + 1
                    ygt= ygt+1
                    yhattarget+=1
                    acc = accuracy_score(ygt.flatten(), yhat.flatten())
                    acctarget= accuracy_score(ygt.flatten(), yhattarget.flatten())
                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g}\n".format(output_loss,mse_loss, total_loss, acc, acctarget))
                    return acctarget, yhat, output_loss, mse_loss, total_loss
        
                eval_acc, eval_yhat, eval_output_loss, eval_mse_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                test_acc, test_yhat, test_output_loss, test_mse_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
                start_time = time.time()
                # Loop over number of epochs
                time_lst=[]
                for epoch in range(config.training_epoch):
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:
                        # Get a batch
                        t1=time.time()
                        (x_batch,y_batch)=train_generator[step]
                        x_batch=np.append(x_batch,np.zeros(x_batch.shape),axis=-1)
                        (x_batch2,y_batch2)=retrain_generator[step]
                        target_bool=np.append(np.ones(len(x_batch2)),np.zeros(len(x_batch)))

                        x_batch0=np.vstack([x_batch2,x_batch])
                        y_batch0=np.vstack([y_batch2,y_batch])
                        t2=time.time()
                        time_lst.append(t2-t1)                        
        
                        train_step_, train_output_loss_, train_mse_loss_, train_total_loss_, train_acc_ = train_step(x_batch0, y_batch0, target_bool)
                        time_str = datetime.now().isoformat()
        
                        print("{}: step {}, output_loss {}, mse_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_mse_loss_, train_total_loss_, train_acc_))
                        step += 1
        
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            eval_acc, eval_yhat, eval_output_loss, eval_mse_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                            test_acc, test_yhat, test_output_loss, test_mse_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
        
                            if(eval_acc > best_acc):
                                best_acc = eval_acc
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
                    retrain_generator.on_epoch_end()
                end_time = time.time()
                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                    text_file.write("{:g}\n".format((end_time - start_time)))
                    text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
                save_neuralnetworkinfo(checkpoint_path, 'fmandclassnetwork',arnn,  originpath=__file__, readme_text=
                        'Feature matching and classification network on cEEGrid (with normalization per patient) \n source net and target net are different, initialized with SeqSleepNet  \n'+
                        'training on {:d} patients \n validation with mse loss instead of accuracy \n no batch norm \n baseline net is trained on 190 pat \n target labels used with separate target classifier, early stop at best eval acc on target. LR 1e-4, mse weight 1.0 \n'.format(number_patients)+
                        print_instance_attributes(config))
