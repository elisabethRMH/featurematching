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
from DenseTranspose import DenseTransp
sys.path.insert(1,'/users/sista/ehereman/GitHub/gradient_reversal_keras_tf')
from flipGradientTF import GradientReversal #24/08/20 different implementation of flip layer. check if same
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

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
        
        ##Inputs
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        
        self.config = config
        
        if not config.adapt_featextractor:
            self.input_x0 = tf.placeholder(tf.float32, [None, 128], name="input_x0")
            self.input_x=self.input_x0
        else:
            self.input_x0=tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, 3], name="input_x0")
            self.input_x=arnn_featureextractor(self.config, self.input_x0, self.dropout_keep_prob_rnn, self.frame_seq_len, reuse=False)
           
        self.input_y = tf.placeholder(tf.float32, [None, 128], name="input_y") #C34 data matched with EOG
        self.label= tf.placeholder(tf.int64, [None,], name="label")
        
        self.unmatched_input_y= tf.placeholder(tf.float32, [None, 128], name='unmatched_input_y') #Extra C34 data
        self.training=tf.placeholder(tf.float32, [],name='training')
        ##Settings
        self.bothdirections = self.config.bothdirections
        self.cycle=self.config.cycle
        self.densetranspose=self.config.densetranspose
        self.unmatched_data=self.config.unmatched_c34data
        self.withclassification=self.config.withclassification

        with tf.device('/gpu:0'), tf.variable_scope('autoencoder'):
            model=tf.keras.models.Sequential()
            l1=Dense(40,
                            name="hiddenlayer",
                            activation='relu')
            l2=Dense(128,
                            name="outputlayer",
                            activation=None)
            model.add(l1)
            model.add(l2)
            self.outputlayer= model(self.input_x)

        if self.cycle:
            with tf.device('/gpu:0'), tf.variable_scope('autoencoder_cycle'):
                model2=tf.keras.models.Sequential()
                if self.densetranspose:
                    l3=DenseTransp(l2,activation='relu',name='hiddenlayerc')
                    l4=DenseTransp(l1, activation=None, name= 'outputlayerc')
                else:
                    l3=Dense(40,
                                name="hiddenlayerc",
                                activation='relu')
                    l4=Dense(128,
                                name="outputlayerc",
                                activation=None)
                model2.add(l3)
                model2.add(l4)
                self.outputlayer_c= model2(self.outputlayer)

            if self.bothdirections:
                self.outputlayer2=model2(self.input_y)
                self.outputlayer_c2=model(self.outputlayer2)
            if self.unmatched_data:
                unmatchedtmp=tf.concat([self.input_y,self.unmatched_input_y],axis=0)
                self.unmatched_output = model(model2(unmatchedtmp))
                tf.debugging.assert_equal(self.unmatched_output,l2(l1(l4(l3(self.unmatched_input_y)))))
        
        if self.withclassification:                    
            with tf.device('/gpu:0'), tf.variable_scope("labelpredictor_net"):
                self.features1= tf.concat([l3(self.unmatched_input_y), l1(l4(l3(self.unmatched_input_y)))], axis=1)
                self.score = fc(self.features1,
                                self.features1.shape[1],
                                self.config.nclass,
                                name="outputC",
                                relu=False)
                self.prediction = tf.argmax(self.score, 1, name="prediction")
                correct_predictions = tf.equal(self.prediction, self.label)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                    
        # calculate cross-entropy output loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss, self.output_loss_mean = output_loss(self.outputlayer, self.input_y)
            if self.cycle:
                self.cycle_loss, self.cycle_loss_mean = output_loss(self.outputlayer_c, self.input_x)
                if self.bothdirections:
                    self.output_loss2, self.output_loss_mean2 = output_loss(self.outputlayer2, self.input_x)
                    self.cycle_loss2, self.cycle_loss_mean2 = output_loss(self.outputlayer_c2, self.input_y)
                if self.unmatched_data:
                    
                    self.cycle_loss3, self.cycle_loss_mean3 = output_loss(self.unmatched_output, unmatchedtmp)
                    self.cycle_loss3= tf.math.scalar_mul(self.training, self.cycle_loss3)
                    self.cycle_loss_mean3= tf.math.scalar_mul(self.training, self.cycle_loss_mean3)
                    
                    if self.withclassification:
                        self.class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.score)
                        self.class_loss_mean = tf.reduce_mean(self.class_loss)
                        
                    
        self.session=session

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            varss   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in varss])
                    #and v not in except_vars_DA])
            self.loss = self.output_loss_mean + self.config.l2_reg_lambda*l2_loss
            if self.cycle:
                self.loss = self.loss+ self.config.cycle_weight * self.cycle_loss_mean
                if self.bothdirections:
                    self.loss = self.loss+ self.output_loss_mean2
                    if not self.unmatched_data:
                        self.loss = self.loss+ self.config.cycle_weight*self.cycle_loss_mean2    
                if self.unmatched_data:
                    self.loss = self.loss+self.config.cycle_weight*self.cycle_loss_mean3
                    if self.withclassification:
                        self.loss = self.loss + 0.1 * self.class_loss_mean

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



