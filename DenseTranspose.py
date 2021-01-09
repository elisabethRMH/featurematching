#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:23:28 2021

@author: ehereman
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

class DenseTransp(Layer):
    def __init__(self, or_layer, activation=None,**kwargs):
        super(DenseTransp, self).__init__(**kwargs)
        self.original_layer=or_layer
        self.activation= keras.activations.get(activation)
    def build(self, batch_inp_shape):
        self.bias = self.add_weight(name='bias', 
                                      shape=[self.original_layer.get_weights()[0].shape[-2]],
                                      initializer='zeros')
        super().build(batch_inp_shape)
        
    def call(self, input_1):
        weights= self.original_layer.get_weights()[0]
        weights = tf.transpose(weights)
        output = tf.linalg.matmul(input_1, weights) + self.bias
        return self.activation(output)