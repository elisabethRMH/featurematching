#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrogram Mapping Network
TODO should maybe add some dropout regularization cause lots of coefficients
Created on Fri Oct 23 13:08:56 2020

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


class MappingNetwork_spectro(object):
    
    def __init__(self, config, session):
        self.out_path=config.out_path
        self.checkpoint_path = config.checkpoint_path
        
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None, 29*129], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 29*129], name="input_y")
        self.training=tf.placeholder(tf.bool,[],name='training')
        
        with tf.device('/gpu:0'), tf.variable_scope('autoencoder'):
            dropoutl= tf.keras.layers.Dropout(1-self.config.dropout_keep_prob_rnn, input_shape=(29*129,))
            self.inputs1=dropoutl(self.input_x, training=self.training)
            self.layer1 = fc(self.inputs1,
                            29*129,
                            300,
                            name="hiddenlayer1",
                            relu=True)
            dropoutl2= tf.keras.layers.Dropout(1-self.config.dropout_keep_prob_rnn, input_shape=(300,))
            self.inputs2=dropoutl2(self.layer1, training=self.training)
            self.layer2 = tf.squeeze(fc(self.inputs2,
                            300,
                            40,
                            name="hiddenlayer2",
                            relu=True))
            dropoutl3= tf.keras.layers.Dropout(1-self.config.dropout_keep_prob_rnn, input_shape=(40,))
            self.inputs3=dropoutl3(self.layer2, training=self.training)
            self.layer2inv = tf.squeeze(fc(self.inputs3,
                            40,
                            300,
                            name="hiddenlayer2inv",
                            relu=True))
            dropoutl4= tf.keras.layers.Dropout(1-self.config.dropout_keep_prob_rnn, input_shape=(300,))
            self.inputs4=dropoutl4(self.layer2inv, training=self.training)
            
            self.outputlayer = tf.squeeze(fc(self.inputs4,
                            300,
                            29*129,
                            name="outputlayer",
                            relu=False))
#            self.outputlayer = tf.squeeze(fc(self.input_x,
#                            128,
#                            128,
#                            name="outputlayer",
#                            relu=False))

        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss, self.output_loss_mean = output_loss(self.outputlayer, self.input_y)
                
        self.session=session

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
                    #and v not in except_vars_DA])
            self.loss = self.output_loss_mean + self.config.l2_reg_lambda*l2_loss

    def evaluate(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch
        }
        output_loss, total_loss,features = self.session.run(
               [self.output_loss_mean, self.loss, self.outputlayer], feed_dict)
        return output_loss, total_loss, features
    


def output_loss(inputs, outputs):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)
    return loss, tf.reduce_mean(loss)


