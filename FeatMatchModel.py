#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:40:01 2020

@author: ehereman
"""

import tensorflow as tf
from arnn_sleep_featureextractor import arnn_featureextractor #V2 is without the fc layer!
from nn_basic_layers import fc

class FeatMatch_Model(object):
    '''Feature mapping and classification model using arnn_featureextractor as a feature extractor network, and adds a classification layer to that
    The loss function consists of the feature map loss and classification loss
    '''
    
    def __init__(self, config):
        self.config=config
#        self.net2active=tf.placeholder(tf.float32,[None])
        self.input_x = tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 3], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None,], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        
        #self.input_x=tf.concat([self.input_x1, self.input_x2],axis=0)
        #self.input_y=tf.concat([self.input_y1, self.input_y2],axis=0)
        #with tf.device('/gpu:0'), tf.variable_scope("arnn2"):
#        with tf.device('/gpu:0'), tf.variable_scope("arnn_eog"):
        self.features2 =arnn_featureextractor(self.config, self.input_x, self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False) #number=1
        if config.same_network:
            self.input_x0= tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 3], name="input_x0")
            self.features1=arnn_featureextractor(self.config, self.input_x0, self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=True)
        else:
            self.features1=tf.placeholder(tf.float32, [None, self.config.nhidden1*2])
          
        with tf.device('/gpu:0'), tf.variable_scope("labelpredictor_net"):
            self.score = fc(self.features1,
                            self.config.nhidden1 * 2,
                            self.config.nclass,
                            name="outputC",
                            relu=False)
            self.prediction = tf.argmax(self.score, 1, name="prediction")

        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            _, self.mse_loss = mse_loss(self.features1, self.features2)
            if config.same_network:
                self.output_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.input_y)
                self.output_loss = tf.reduce_sum(self.output_loss)
            else:
                self.output_loss=tf.zeros([], tf.float32)
            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eeg')
            except_vars_eog2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eog')
            except_vars_emg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg2 and v not in except_vars_eog2 and v not in except_vars_emg2])
            self.loss = self.config.l2_reg_lambda*l2_loss +self.mse_loss + self.output_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            

def mse_loss(inputs, outputs):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)

    return loss, tf.reduce_sum(loss)#*total_count1/total_count2


        