#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature matching and classification model with ARNN as feature extractor

Created on Tue May 26 16:40:01 2020

@author: ehereman
"""
import sys
import tensorflow as tf
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/E2E-ARNN')
from nn_basic_layers import fc
from arnn_sleep_featureextractor import arnn_featureextractor #V2 is without the fc layer!
import copy
import tensorflow.keras.backend as K

class FMandClass_Model(object):
    '''Feature mapping and classification model using arnn_featureextractor as a feature extractor network, and adds a classification layer to that
    The loss function consists of the feature map loss and classification loss
    '''
    
    def __init__(self, config):
        self.config=config
        self.net1active=tf.placeholder(tf.float32,[None])
        self.net2active=tf.placeholder(tf.float32,[None])
        self.input_x = tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 2], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.nclass], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.training=tf.placeholder(tf.bool, name='training')

        if config.feature_extractor:
            with tf.device('/gpu:0'), tf.variable_scope("arnn_source"):
                self.features1 =arnn_featureextractor(self.config,self.input_x[:,:,:,0:1], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False, istraining=self.training) #number=1
            
                if config.same_network:
                    self.features2 = arnn_featureextractor(self.config, self.input_x[:,:,:,1:2], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=True, istraining = self.training)
            if not config.same_network:    
                with tf.device('/gpu:0'), tf.variable_scope("arnn_target"):
                    self.features2 =arnn_featureextractor(self.config, self.input_x[:,:,:,1:2], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False, istraining = self.training) #number=1

        else:
            self.features1 =tf.placeholder(tf.float32, [None, self.config.nhidden1*2])
            self.features2 =tf.placeholder(tf.float32, [None, self.config.nhidden1*2])
            
        with tf.device('/gpu:0'), tf.variable_scope("labelpredictor_net"):
            self.score = fc(self.features1,
                            self.config.nhidden1 * 2,
                            self.config.nclass,
                            name="outputsource",
                            relu=False)
            self.prediction = tf.argmax(self.score, 1, name="prediction")
            self.score_target = fc(self.features2,
                            self.config.nhidden1 * 2,
                            self.config.nclass,
                            name="outputtarget",
                            relu=False, reuse=False)
#            self.score_target = fc(self.features2, 
#                            self.config.nhidden1 * 2,
#                            self.config.nclass,
#                            name="outputC",
#                            relu=False, reuse=True)
            self.prediction_target = tf.argmax(self.score_target,1, name='predictiontarget')

        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score)
            
            #addded for version 8 on 31/01/'21
            self.output_loss1 = tf.math.multiply(self.net1active, self.output_loss)
            self.output_loss2 = tf.math.multiply(self.net2active, self.output_loss)
            
            self.output_loss1 = tf.reduce_sum(self.output_loss1)
            self.output_loss2 = tf.reduce_sum(self.output_loss2)
            _, self.mse_loss = mse_loss(self.features1, self.features2,self.net2active)
            
            if self.config.withtargetclass:
                self.output_loss_target=  tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score_target)
                self.output_loss_target = tf.math.multiply(self.net2active, self.output_loss_target)
                self.output_loss_target = tf.reduce_sum(self.output_loss_target)
                
#                self.output_loss2 = tf.math.multiply(self.net2active, self.output_loss)
#                self.output_loss2 = tf.reduce_sum(self.output_loss2)

        #Sum losses with L2 regularization                                 
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_source/filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_source/filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_source/filterbank-layer-emg')
            except_vars_eeg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_target/filterbank-layer-eeg')
            except_vars_eog2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_target/filterbank-layer-eog')
            except_vars_emg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_target/filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg
                    and v not in except_vars_eeg2 and v not in except_vars_eog2 and v not in except_vars_emg2])
            self.loss = self.output_loss1 +  self.config.l2_reg_lambda*l2_loss + self.config.mse_weight* self.mse_loss
            if self.config.withtargetclass:
                self.loss+=1.0*(self.output_loss_target) #+self.output_loss2
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            

def mse_loss(inputs, outputs,net2active):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)
    loss = tf.math.multiply(net2active, loss)
    # total_count1 = tf.to_float(tf.shape(loss)[0])
    # total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(net2active, tf.int32))) #V2 adaptation Elisabeth 11/08/'20

    return loss, tf.reduce_sum(loss)#*total_count1/total_count2

def mmd_loss(inputs, outputs,beta=0.2):
    return tf.reduce_mean(gaussian_kernel(inputs,inputs,beta))+ tf.reduce_mean(gaussian_kernel(outputs,outputs,beta))- 2*tf.reduce_mean(gaussian_kernel(inputs,outputs,beta))
                        

def gaussian_kernel(x1, x2, beta = 0.2):
    r = tf.expand_dims(x1, 1)
    return K.exp( -beta * tf.math.reduce_sum(K.square(r - x2), axis=-1))  

