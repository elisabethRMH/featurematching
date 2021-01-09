#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:08:56 2020

@author: ehereman
"""
import tensorflow as tf
import sys
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/E2E-ARNN')
from nn_basic_layers import bidirectional_recurrent_layer,bidirectional_recurrent_layer_output, attention, fc
from filterbank_shape import FilterbankShape

sys.path.insert(1,'/users/sista/ehereman/GitHub/gradient_reversal_keras_tf')
from flipGradientTF import GradientReversal #24/08/20 different implementation of flip layer. check if same

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


class AdversarialNetworkF(object):
    
    def __init__(self, config, session):
        self.out_path=config.out_path
        self.checkpoint_path = config.checkpoint_path
        
        self.config = config
        self.domain_lambda= self.config.domain_lambda#tf.constant()
#        self.input_x = tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 3], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
#        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.feat_input=tf.placeholder(tf.float32, [None, 128],name='feat_input')

#        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN

#        filtershape = FilterbankShape()
#        #triangular filterbank
#        self.Wbl = tf.constant(filtershape.lin_tri_filter_shape(nfilt=self.config.nfilter,
#                                                                nfft=self.config.nfft,
#                                                                samplerate=self.config.samplerate,
#                                                                lowfreq=self.config.lowfreq,
#                                                                highfreq=self.config.highfreq),
#                               dtype=tf.float32,
#                               name="W-filter-shape-eeg")
#
#        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eeg"):
#            Xeeg = tf.reshape(tf.squeeze(self.input_x[:,:,:,0]), [-1, self.config.ndim])
#            # first filter bank layer
#            self.Weeg = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
#            # non-negative constraints
#            self.Weeg = tf.sigmoid(self.Weeg)
#            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
#            self.Wfb = tf.multiply(self.Weeg,self.Wbl)
#            HWeeg = tf.matmul(Xeeg, self.Wfb) # filtering
#            HWeeg = tf.reshape(HWeeg, [-1, self.config.frame_step, self.config.nfilter])
#
#        #assert(self.config.nchannel==1)
#        if(self.config.nchannel > 1):
#            with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eog"):
#                # Temporarily crush the feature_mat's dimensions
#                Xeog = tf.reshape(tf.squeeze(self.input_x[:,:,:,1]), [-1, self.config.ndim])
#                # first filter bank layer
#                self.Weog = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
#                # non-negative constraints
#                self.Weog = tf.sigmoid(self.Weog)
#                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
#                self.Wfb = tf.multiply(self.Weog,self.Wbl)
#                HWeog = tf.matmul(Xeog, self.Wfb) # filtering
#                HWeog = tf.reshape(HWeog, [-1, self.config.frame_step, self.config.nfilter])
#
#        if(self.config.nchannel > 2):
#            with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-emg"):
#                # Temporarily crush the feature_mat's dimensions
#                Xemg = tf.reshape(tf.squeeze(self.input_x[:,:,:,2]), [-1, self.config.ndim])
#                # first filter bank layer
#                self.Wemg = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
#                # non-negative constraints
#                self.Wemg = tf.sigmoid(self.Wemg)
#                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
#                self.Wfb = tf.multiply(self.Weog,self.Wbl)
#                HWemg = tf.matmul(Xemg, self.Wfb) # filtering
#                HWemg = tf.reshape(HWemg, [-1, self.config.frame_step, self.config.nfilter])
#
#        if(self.config.nchannel > 2):
#            X = tf.concat([HWeeg, HWeog, HWemg], axis = 2)
#        elif(self.config.nchannel > 1):
#            X = tf.concat([HWeeg, HWeog], axis = 2)
#        else:
#            X = HWeeg
#
#        # bidirectional frame-level recurrent layer
#        with tf.device('/gpu:0'), tf.variable_scope("frame_rnn_layer") as scope:
#            fw_cell1, bw_cell1 = bidirectional_recurrent_layer(self.config.nhidden1,
#                                                                  self.config.nlayer1,
#                                                                  input_keep_prob=self.dropout_keep_prob_rnn,
#                                                                  output_keep_prob=self.dropout_keep_prob_rnn)
#            rnn_out1, rnn_state1 = bidirectional_recurrent_layer_output(fw_cell1,
#                                                                        bw_cell1,
#                                                                        X,
#                                                                        self.frame_seq_len,
#                                                                        scope=scope)
#            print(rnn_out1.get_shape())
#            # output shape (batchsize*epoch_step, frame_step, nhidden1*2)
#
#        with tf.device('/gpu:0'), tf.variable_scope("frame_attention_layer"):
#            self.attention_out1 = attention(rnn_out1, self.config.attention_size1)
#            print(self.attention_out1.get_shape())            
#            #assert_shape(self.attention_out1, [None, 128])
#            assert(self.attention_out1.get_shape().as_list()==[None, 128])
        
        with tf.device('/gpu:0'), tf.variable_scope('labelpredictor_net'):
            self.score_C = fc(self.feat_input,
                                self.config.nhidden1 * 2,
                                self.config.nclass,
                                name="outputC",
                                relu=False)
            self.predictionC = tf.argmax(self.score_C, 1, name="prediction")
        
        if config.domainclassifier or config.domainclassifierstage2:
            with tf.device('/gpu:0'), tf.variable_scope('domainclassifier_net'):
                #self.domain_x1= flip_gradient(self.attention_out1) 
                fliplayer= GradientReversal(1.0)
                self.domain_x1 = fliplayer(self.attention_out1)
                if config.add_classifieroutput:
                    self.domain_x0 = tf.stop_gradient(self.score_C)
                    self.domain_x1= tf.concat([self.domain_x1, self.domain_x0],1)
                if config.add_classifieroutput:
                    self.domain_x2 = fc(self.domain_x1,
                                    self.config.nhidden1 * 2 +self.config.nclass,
                                    self.config.nfc_domainclass,
                                    name="domain1",
                                    relu=True)
                else:                    
                    self.domain_x2 = fc(self.domain_x1,
                                    self.config.nhidden1 * 2,
                                    self.config.nfc_domainclass,
                                    name="domain1",
                                    relu=True)
#                self.domain_x3 = fc(self.domain_x2,
#                                self.config.nfc_domainclass,
#                                self.config.nfc_domainclass,
#                                name="domain2",
#                                relu=True)
                self.score_D = tf.squeeze(fc(self.domain_x2,
                                self.config.nfc_domainclass,
                                1,
                                name="outputD",
                                relu=False))
                self.predictionD = tf.math.sigmoid(self.score_D, name="prediction")
                result=self.predictionD>0.5
                applicable = tf.not_equal(self.input_y, -1)
                self.accuracyD=tf.reduce_mean(tf.to_float( tf.equal(result, applicable)))
        else:
            self.predictionD= self.predictionC*0
            self.accuracyD = tf.constant(0)
        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss_mean, self.output_loss = classification_costs(labels=self.input_y, logits=self.score_C)
            self.accuracy, _= accuracy(labels=self.input_y, logits = self.score_C)
            
            if config.domainclassifier:
                self.domain_loss_mean, self.domain_loss= domainclassification_costs(labels= self.input_y, logits= self.score_D)            
                self.loss_unreg= self.output_loss_mean+self.domain_lambda* self.domain_loss_mean#tf.reduce_mean(self.output_loss+ (self.domain_lambda) * self.domain_loss)
            elif config.domainclassifierstage2:
                self.domain_loss_mean, self.domain_loss= domainclassification_costs(labels= self.input_y, logits= self.score_D)            
                self.loss_unreg= self.domain_lambda* self.domain_loss_mean#tf.reduce_mean(self.output_loss+ (self.domain_lambda) * self.domain_loss)                
            else:
                self.domain_loss_mean=tf.constant(0)
                self.domain_loss=tf.zeros_like(self.output_loss)
                self.loss_unreg= self.output_loss_mean #tf.reduce_mean(self.output_loss)
                
        self.session=session

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-emg')
            except_vars_DA= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='domainclassifier_net')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
                    #and v not in except_vars_DA])
            self.loss = self.loss_unreg + self.config.l2_reg_lambda*l2_loss

            

        
    
    def initialize(self,fromFullySup=False ,checkpoint=None):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        print("Writing to {}\n".format(self.out_path))

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        
        # initialize all variables
        
        if checkpoint is None:
            self.session.run(tf.initialize_all_variables())
            print("Model initialized")
        elif not fromFullySup:
            # Load saved model to continue training or initialize all variables
            best_dir = os.path.join(checkpoint, "best_model_acc")
            self.saver.restore(self.session, best_dir)
            print("Model loaded")
        else:
            variables= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eeg')
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eog'))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-emg'))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='frame_rnn_layer'))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='frame_attention_layer'))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='labelpredictor_net'))
            saver_fullySup= tf.train.Saver(variables)
            self.session.run(tf.initialize_all_variables())
            best_dir = os.path.join(checkpoint, "best_model_acc")
            saver_fullySup.restore(self.session, best_dir)
        

    def train(self, train_generator, eval_generator, test_generator, training_epoch, training_length):
        # variable to keep track of best fscore
        train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
        best_fscore = 0.0
        best_acc = 0.0
        best_kappa = 0.0
        min_loss = float("inf")

        # Loop over number of epochs
        global_step=0
        epoch=0
        while global_step< training_length:
            epoch=epoch+1
#        for epoch in range(training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 0
            while step < train_batches_per_epoch:
                # Get a batch
                (x_batch,y_batch) = train_generator[step]
                x_batch=x_batch[:,0,:,:,0:3]
                y_batch=y_batch[:,0]

                train_step_, train_output_loss_, train_domain_loss_, train_total_loss_, train_acc_, pred_D, train_acc_D = self.train_step(x_batch, y_batch)
                time_str = datetime.now().isoformat()

                print("{}: step {}, output_loss {}, domain_loss {}, total_loss {} acc {} accD {} predD {}".format(time_str, train_step_, train_output_loss_, train_domain_loss_, train_total_loss_, train_acc_, train_acc_D, np.sum(pred_D<0.5)))
                step += 1

                current_step = tf.train.global_step(self.session, self.global_step)
                if current_step % self.config.evaluate_every == 0:
                    # Validate the model on the entire evaluation test set after each epoch
                    print("{} Start validation".format(datetime.now()))
                    eval_acc, eval_yhat, eval_output_loss, eval_domain_loss, eval_total_loss = self.evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                    test_acc, test_yhat, test_output_loss, test_domain_loss, test_total_loss = self.evaluate(gen=test_generator, log_filename="test_result_log.txt")

                    if(eval_acc >= best_acc):
                        best_acc = eval_acc
                        checkpoint_name = os.path.join(self.checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                        save_path = self.saver.save(self.session, checkpoint_name)
                        
                        
                        print("Best model updated")
                        source_file = checkpoint_name
                        dest_file = os.path.join(self.checkpoint_path, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')

            train_generator.on_epoch_end()
            global_step+=train_batches_per_epoch

    def train_step(self, x_batch, y_batch): #not adapted
        """
        A single training step
        """
        frame_seq_len = np.ones(len(x_batch),dtype=int) * self.config.frame_seq_len
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob_rnn: self.config.dropout_keep_prob_rnn,
          self.frame_seq_len: frame_seq_len
        }
        _, step, output_loss, domain_loss, total_loss, accuracy, predD,accuracyD = self.session.run(
           [self.train_op, self.global_step, self.output_loss_mean, self.domain_loss_mean, self.loss, self.accuracy, self.predictionD, self.accuracyD],
           feed_dict)
        return step, output_loss, domain_loss, total_loss, accuracy, predD, accuracyD
    
    def dev_step(self, x_batch, y_batch):
        frame_seq_len = np.ones(len(x_batch),dtype=int) * self.config.frame_seq_len
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob_rnn: 1.0,
            self.frame_seq_len: frame_seq_len
        }
        output_loss, domain_loss, total_loss, yhat,predD  = self.session.run(
               [self.output_loss_mean,self.domain_loss_mean, self.loss, self.predictionC, self.predictionD], feed_dict)
        return output_loss, domain_loss, total_loss, yhat

    def evalfeatures_adversarialnets(self, x_batch, y_batch):
        feed_dict = {
            self.feat_input: x_batch,
            self.input_y: y_batch
        }
        output_loss, domain_loss, total_loss, yhat, score, features = self.session.run(
               [self.output_loss_mean,self.domain_loss_mean, self.loss, self.predictionC, self.score_C, self.feat_input], feed_dict)
        return output_loss, total_loss, yhat, score, features
    
    def evaluate(self, gen, log_filename):
        # Validate the model on the entire evaluation test set after each epoch
    
        output_loss =0
        domain_loss=0
        total_loss = 0
        yhat = np.zeros(len(gen.datalist))
        num_batch_per_epoch = len(gen)
        test_step = 0
        ygt = np.zeros(len(gen.datalist))
        while test_step < num_batch_per_epoch:
            #((x_batch, y_batch),_) = gen[test_step]
            (x_batch,y_batch)=gen[test_step]
            x_batch=x_batch[:,0,:,:,0:3]
            #x_batch=x_batch[:,0]
            y_batch=y_batch[:,0]
    
            output_loss_, domain_loss_, total_loss_, yhat_ = self.dev_step(x_batch, y_batch)
            output_loss += output_loss_
            total_loss += total_loss_
            domain_loss += domain_loss_
    
            yhat[(test_step)*self.config.batch_size : (test_step+1)*self.config.batch_size] = yhat_
            ygt[(test_step)*self.config.batch_size : (test_step+1)*self.config.batch_size] = y_batch
            test_step += 1
        if len(gen.datalist) - test_step*self.config.batch_size==1:
            yhat=yhat[0:-1]
            ygt=ygt[0:-1]
                
        elif len(gen.datalist) > test_step*self.config.batch_size:
            # if using load_random_tuple
            #((x_batch, y_batch),_) = gen[test_step]
            (x_batch,y_batch)=gen[test_step]
            x_batch=x_batch[:,0,:,:,0:3]
            y_batch=y_batch[:,0]
    
            output_loss_, domain_loss_, total_loss_, yhat_ = self.dev_step(x_batch, y_batch)
            ygt[(test_step)*self.config.batch_size : len(gen.datalist)] = y_batch
            yhat[(test_step)*self.config.batch_size : len(gen.datalist)] = yhat_
            output_loss += output_loss_
            total_loss += total_loss_
            domain_loss += domain_loss_
        yhat = yhat + 1
        ygt= ygt+1
        acc = accuracy_score(ygt, yhat)
        with open(os.path.join(self.out_path, log_filename), "a") as text_file:
            text_file.write("{:g} + {:g} = {:g}, acc: {:g}\n".format(output_loss, domain_loss, total_loss, acc))
        return acc, yhat, output_loss, domain_loss, total_loss


        
def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count1 = tf.to_float(tf.shape(per_sample)[0])
        total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(applicable, tf.int32))) #V2 adaptation Elisabeth 11/08/'20
        #mean = tf.div(labeled_sum, total_count2, name=scope)
        labeled_sum = tf.multiply(labeled_sum, total_count1/total_count2, name=scope)
        return labeled_sum, per_sample#mean, per_sample

def domainclassification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "domainclass_costs") as scope:
        applicable = tf.not_equal(labels, -1)
        total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(applicable, tf.int32))) #V2 adaptation Elisabeth 11/08/'20

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, tf.ones_like(labels), tf.zeros_like(labels))
        labels = tf.cast(labels, tf.float32)
        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        total_count1 = tf.to_float(tf.shape(per_sample)[0])

        # Take mean over all examples, not just labeled examples.
        per_sample = tf.where(applicable, per_sample*(total_count1)/total_count2/2, per_sample*(total_count1)/(total_count1-total_count2)/2)
        labeled_sum = tf.reduce_sum(per_sample)
        #total_count = tf.to_float(tf.shape(per_sample)[0])
        #mean = tf.div(labeled_sum, total_count, name=scope)

        return labeled_sum, per_sample#mean, per_sample

def accuracy(logits, labels, name=None):
    """Compute accuracy mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean acc over unlabeled examples.
    Mean acc is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "accs") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample

#def flip_gradient(x, l=1.0):
#    'copied from https://github.com/tachitachi/GradientReversal '
#    positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
#    negative_path = -x * tf.cast(l, tf.float32)
#    return positive_path + negative_path        
