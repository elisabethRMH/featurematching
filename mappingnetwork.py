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
from arnn_sleep_featureextractor import arnn_featureextractor #V2 is without the fc layer!


class MappingNetwork(object):
    
    def __init__(self, config, session):
        self.out_path=config.out_path
        self.checkpoint_path = config.checkpoint_path
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        
        self.config = config
        
        if not config.adapt_featextractor:
            self.input_x0 = tf.placeholder(tf.float32, [None, 128], name="input_x0")
            self.input_x=self.input_x0
        else:
            self.input_x0=tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 3], name="input_x0")
            self.input_x=arnn_featureextractor(self.config, self.input_x0, self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False)
           
        self.input_y = tf.placeholder(tf.float32, [None, 128], name="input_y")
        self.EOGtoC34 = True
        self.cycle=self.config.cycle
        
        with tf.device('/gpu:0'), tf.variable_scope('autoencoder'):
            
            self.hiddenlayer = fc(self.input_x,
                            128,
                            40,
                            name="hiddenlayer",
                            relu=True)
#            self.layer2 = tf.squeeze(fc(self.layer1,
#                            40,
#                            20,
#                            name="hiddenlayer2",
#                            relu=True))
#            self.layer2inv = tf.squeeze(fc(self.layer2,
#                            20,
#                            40,
#                            name="hiddenlayer2inv",
#                            relu=True))
            
            self.outputlayer = tf.squeeze(fc(self.hiddenlayer,
                            40,
                            128,
                            name="outputlayer",
                            relu=False))
        if self.cycle:
            with tf.device('/gpu:0'), tf.variable_scope('autoencoder_cycle'):
                self.hiddenlayer_c = fc(self.outputlayer,
                                128,
                                40,
                                name="hiddenlayer_c",
                                relu=True)
    #            self.layer2 = tf.squeeze(fc(self.layer1,
    #                            40,
    #                            20,
    #                            name="hiddenlayer2",
    #                            relu=True))
    #            self.layer2inv = tf.squeeze(fc(self.layer2,
    #                            20,
    #                            40,
    #                            name="hiddenlayer2inv",
    #                            relu=True))
                
                self.outputlayer_c = tf.squeeze(fc(self.hiddenlayer_c,
                                40,
                                128,
                                name="outputlayer_c",
                                relu=False))
        
        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss, self.output_loss_mean = output_loss(self.outputlayer, self.input_y)
            if self.cycle:
                self.cycle_loss, self.cycle_loss_mean = output_loss(self.outputlayer_c, self.input_x)
        self.session=session

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            varss   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in varss])
                    #and v not in except_vars_DA])
            self.loss = self.output_loss_mean + self.config.l2_reg_lambda*l2_loss
            if self.cycle:
                self.loss = self.loss+ self.config.cycle_weight * self.cycle_loss_mean

    def evaluate(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch
        }
        output_loss, total_loss,before_mapping_features, after_mapping_features = self.session.run(
               [self.output_loss_mean, self.loss, self.input_x, self.outputlayer], feed_dict)
        return output_loss, total_loss, before_mapping_features, after_mapping_features
    


def output_loss(inputs, outputs):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)
    return loss, tf.reduce_mean(loss)



