#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature matching and classification model with SeqSleepNet as feature extractor

Created on Tue May 26 16:40:01 2020

@author: ehereman
"""
import sys
import tensorflow as tf
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet')
from nn_basic_layers import fc
from seqsleepnet_sleep_featureextractor import seqsleepnet_featureextractor #V2 is without the fc layer!
from seqsleepnet_sleep_featureextractor_diffattn import seqsleepnet_featureextractor_diffattn
import copy
import tensorflow.keras.backend as K

class FMandClass_ModelSeqSlNet(object):
    '''Feature mapping and classification model using seqsleepnet_featureextractor as a feature extractor network, and adds a classification layer to that
    The loss function consists of the feature map loss and classification loss
    '''
    
    def __init__(self, config):
        self.config=config
        self.net1active=tf.placeholder(tf.float32,[None])
        self.net2active=tf.placeholder(tf.float32,[None])
        self.input_x = tf.compat.v1.placeholder(tf.float32, [None, self.config.epoch_step, self.config.frame_step, self.config.ndim, 2], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.config.epoch_step, self.config.nclass], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.training=tf.placeholder(tf.bool, name='training')

        if config.feature_extractor:

            with tf.device('/gpu:0'), tf.variable_scope("seqsleepnet_source"):
                self.features1 =seqsleepnet_featureextractor(self.config,self.input_x[:,:,:,:,0:1], self.dropout_keep_prob_rnn, self.frame_seq_len, self.epoch_seq_len, reuse=False, istraining=self.training) #number=1

                tmp= tf.boolean_mask(self.input_x,tf.dtypes.cast(self.net2active, tf.bool))
                frame_tmp= tf.repeat(tf.boolean_mask(self.frame_seq_len,tf.dtypes.cast(self.net2active, tf.bool)),self.config.epoch_seq_len, axis=0)
                epoch_tmp= tf.boolean_mask(self.epoch_seq_len,tf.dtypes.cast(self.net2active, tf.bool))

                if config.same_network:
                    self.features2 =seqsleepnet_featureextractor(self.config, tmp[:,:,:,:,1:2], self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=True, istraining = self.training) #number=1
                    if config.diffattn:
                        reuseattn=not config.diffattn
                        reuseepochrnn= not config.diffepochrnn
                        self.features2= seqsleepnet_featureextractor_diffattn(self.config, tmp[:,:,:,:,1:2], self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=True, istraining = self.training, reuseattn=reuseattn, reuseepochrnn=reuseepochrnn) #number=1
            if not config.same_network:
                with tf.device('/gpu:0'), tf.variable_scope("seqsleepnet_target"):                            
                    self.features2 =seqsleepnet_featureextractor(self.config, tmp[:,:,:,:,1:2], self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=False, istraining = self.training) #number=1

        else:
            self.features1 =tf.placeholder(tf.float32, [None, config.epoch_seq_len, self.config.nhidden2*2])
            self.features2 =tf.placeholder(tf.float32, [None, config.epoch_seq_len, self.config.nhidden2*2])
            
        self.scores = []
        self.predictions = []
        self.scores_target = []
        self.predictions_target = []        
        with tf.device('/gpu:0'), tf.compat.v1.variable_scope("output_layer"):
            for i in range(self.config.epoch_step):
                score_i = fc((self.features1[:,i,:]),
                                self.config.nhidden2 * 2,
                                self.config.nclass,
                                name="output-%s" % i,
                                relu=False) #output: logits without softmax!
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                self.scores.append(score_i)
                self.predictions.append(pred_i)
                score_i_target = fc((self.features2[:,i,:]),
                                self.config.nhidden2 * 2,
                                self.config.nclass,
                                name="outputtarget-%s" % i,
                                relu=False) #output: logits without softmax!
                pred_i_target = tf.argmax(score_i_target, 1, name="predtarget-%s" % i)
                self.scores_target.append(score_i_target)
                self.predictions_target.append(pred_i_target)

        # calculate cross-entropy output loss
        self.output_loss = 0
        self.output_loss2 = 0
        self.output_loss_target = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            for i in range(self.config.epoch_step):
                output_loss_i = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=(self.input_y[:,i,:]), logits=self.scores[i])            
                tmp= tf.boolean_mask(output_loss_i,tf.dtypes.cast(self.net1active, tf.bool))                
                self.output_loss_i = tf.reduce_sum(tmp, axis=[0])
                self.output_loss += self.output_loss_i
                
                tmp2= tf.boolean_mask(output_loss_i,tf.dtypes.cast(self.net2active, tf.bool)) #matched source output
                self.output_loss_i2= tf.reduce_sum(tmp2, axis=[0])
                self.output_loss2+= self.output_loss_i2
                
                if self.config.withtargetclass:                    
                    tmpy= tf.boolean_mask(self.input_y,tf.dtypes.cast(self.net2active, tf.bool))
                    self.output_loss_i_target=  tf.nn.softmax_cross_entropy_with_logits(labels=tmpy[:,i,:], logits=self.scores_target[i])
                    self.output_loss_i_target = tf.reduce_sum(self.output_loss_i_target)
                    self.output_loss_target += self.output_loss_i_target
            self.output_loss_target = self.output_loss_target/ self.config.epoch_step
            self.output_loss = self.output_loss/self.config.epoch_step # average over sequence length
            self.output_loss2 = self.output_loss2/self.config.epoch_step # average over sequence length

            
            tmp= tf.boolean_mask(self.features1,tf.dtypes.cast(self.net2active, tf.bool))
            tmp2= tf.boolean_mask(self.features1,tf.dtypes.cast(self.net1active, tf.bool))
            _, self.mse_loss = mse_loss(tmp, self.features2)
            if self.config.mmd_loss:
                self.mmd_loss = mmd_loss(tmp, self.features2)
                
            self.mse_loss= self.mse_loss/self.config.epoch_step
            if self.config.mmd_loss:
                self.mmd_loss= self.mmd_loss/self.config.epoch_step
                
            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-emg')
            except_vars_eeg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-eeg')
            except_vars_eog2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-eog')
            except_vars_emg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg
                    and v not in except_vars_eeg2 and v not in except_vars_eog2 and v not in except_vars_emg2])
            self.loss =self.output_loss+  self.config.l2_reg_lambda*l2_loss
            if self.config.withtargetclass:
                self.loss+=1.0*(self.output_loss_target) #
            if self.config.mmd_loss:
                self.loss+= self.config.mmd_weight* self.mmd_loss
            if self.config.mse_loss:
                self.loss+= self.config.mse_weight* self.mse_loss

        self.accuracy = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            for i in range(self.config.epoch_step):
                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)


            

def mse_loss(inputs, outputs,netactive=None):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    loss= mse(outputs, inputs)
    if netactive is not None:
        loss = tf.math.multiply(netactive, loss)
#    total_count1 = tf.to_float(tf.shape(loss)[0])
#    total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(netactive, tf.int32))) #V2 adaptation Elisabeth 11/08/'20

    return loss, tf.reduce_sum(loss)#*total_count1/total_count2


def flip_gradient(x, l=1.0):
    'copied from https://github.com/tachitachi/GradientReversal '
    positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
    negative_path = -x * tf.cast(l, tf.float32)
    return positive_path + negative_path   


def mmd_loss(inputs, outputs,beta=0.2):
    return tf.reduce_mean(gaussian_kernel(inputs,inputs,beta))+ tf.reduce_mean(gaussian_kernel(outputs,outputs,beta))- 2*tf.reduce_mean(gaussian_kernel(inputs,outputs,beta))
                        

def gaussian_kernel(x1, x2, beta = 0.2):
    r = tf.expand_dims(x1, 1)
    return K.exp( -beta * tf.math.reduce_sum(K.square(r - x2), axis=-1))  

