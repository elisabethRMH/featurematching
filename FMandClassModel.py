#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:40:01 2020

@author: ehereman
"""

import tensorflow as tf
from arnn_sleep_featureextractor import arnn_featureextractor #V2 is without the fc layer!
from nn_basic_layers import fc

class FMandClass_Model(object):
    '''Feature mapping and classification model using arnn_featureextractor as a feature extractor network, and adds a classification layer to that
    The loss function consists of the feature map loss and classification loss
    '''
    
    def __init__(self, config):
        self.config=config
        self.net2active=tf.placeholder(tf.float32,[None])
        self.input_x = tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 4], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.nclass], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        
        #self.input_x=tf.concat([self.input_x1, self.input_x2],axis=0)
        #self.input_y=tf.concat([self.input_y1, self.input_y2],axis=0)
        if config.feature_extractor:
            if config.same_network:
                self.features1 =arnn_featureextractor(self.config, self.input_x[:,:,:,0:3], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False) #number=1
                self.features2 =arnn_featureextractor(self.config, self.input_x[:,:,:,1:3], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=True) #number=1
            else:
                with tf.device('/gpu:0'), tf.variable_scope("arnn_c34"):
                    self.features1 =arnn_featureextractor(self.config, self.input_x[:,:,:,0:3], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False) #number=1
                #with tf.device('/gpu:0'), tf.variable_scope("arnn2"):
                with tf.device('/gpu:0'), tf.variable_scope("arnn_eog"):
                    self.features2 =arnn_featureextractor(self.config, self.input_x[:,:,:,1:3], self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False) #number=1
        else:
            self.features1 =tf.placeholder(tf.float32, [None, self.config.nhidden1*2])
            self.features2 =tf.placeholder(tf.float32, [None, self.config.nhidden1*2])
            
        with tf.device('/gpu:0'), tf.variable_scope("labelpredictor_net"):
            self.score = fc(self.features1,
                            self.config.nhidden1 * 2,
                            self.config.nclass,
                            name="outputC",
                            relu=False)
            self.prediction = tf.argmax(self.score, 1, name="prediction")

        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score)
            self.output_loss = tf.reduce_sum(self.output_loss)
            _, self.mse_loss = mse_loss(self.features1, self.features2,self.net2active)

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_c34/filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_c34/filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_c34/filterbank-layer-emg')
            except_vars_eeg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_eog/filterbank-layer-eeg')
            except_vars_eog2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_eog/filterbank-layer-eog')
            except_vars_emg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='arnn_eog/filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg
                    and v not in except_vars_eeg2 and v not in except_vars_eog2 and v not in except_vars_emg2])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss + self.config.mse_weight* self.mse_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            

def mse_loss(inputs, outputs,net2active):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)
    loss = tf.math.multiply(net2active, loss)
    total_count1 = tf.to_float(tf.shape(loss)[0])
    total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(net2active, tf.int32))) #V2 adaptation Elisabeth 11/08/'20

    return loss, tf.reduce_sum(loss)#*total_count1/total_count2


        