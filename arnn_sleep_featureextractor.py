import tensorflow as tf
import sys
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/E2E-ARNN')
#sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet')
from nn_basic_layers import *
from filterbank_shape import FilterbankShape
'''
Attentional recurrent neural network for sleep staging.
This code is based on arnn_sleep_selfsup_v2.py from selfsup_banville. 
I adapted it into a function cause that's better than making a class where the init is the only function called.
Also fixed mistake in filter bank for EMG. even though EMG is not being used here
'''

def arnn_featureextractor(config, input_x, dropout_keep_prob_rnn, frame_seq_len, reuse=False, istraining=False):


    filtershape = FilterbankShape()
    #triangular filterbank
    Wbl = tf.constant(filtershape.lin_tri_filter_shape(nfilt=config.nfilter,
                                                            nfft=config.nfft,
                                                            samplerate=config.samplerate,
                                                            lowfreq=config.lowfreq,
                                                            highfreq=config.highfreq),
                           dtype=tf.float32,
                           name="W-filter-shape-eeg")

    with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eeg", reuse=reuse):
        # Temporarily crush the feature_mat's dimensions
        Xeeg = tf.reshape(tf.squeeze(input_x[:,:,:,0]), [-1, config.ndim])
        # first filter bank layer
        Weeg = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
#        Weeg = tf.Variable(tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
        # non-negative constraints
        Weeg = tf.sigmoid(Weeg)
        # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
        Wfb = tf.multiply(Weeg,Wbl)
        HWeeg = tf.matmul(Xeeg, Wfb) # filtering
        HWeeg = tf.reshape(HWeeg, [-1, config.frame_step, config.nfilter])

    if(config.nchannel > 1):
        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eog", reuse=reuse):
            # Temporarily crush the feature_mat's dimensions
            Xeog = tf.reshape(tf.squeeze(input_x[:,:,:,1]), [-1, config.ndim])
            # first filter bank layer
            Weog = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
            # non-negative constraints
            Weog = tf.sigmoid(Weog)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb = tf.multiply(Weog,Wbl)
            HWeog = tf.matmul(Xeog, Wfb) # filtering
            HWeog = tf.reshape(HWeog, [-1, config.frame_step, config.nfilter])

    if(config.nchannel > 2):
        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-emg", reuse=reuse):
            # Temporarily crush the feature_mat's dimensions
            Xemg = tf.reshape(tf.squeeze(input_x[:,:,:,2]), [-1, config.ndim])
            # first filter bank layer
            Wemg = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
            # non-negative constraints
            Wemg = tf.sigmoid(Wemg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb = tf.multiply(Wemg,Wbl)
            HWemg = tf.matmul(Xemg, Wfb) # filtering
            HWemg = tf.reshape(HWemg, [-1, config.frame_step, config.nfilter])

    if(config.nchannel > 2):
        X = tf.concat([HWeeg, HWeog, HWemg], axis = 2)
    elif(config.nchannel > 1):
        X = tf.concat([HWeeg, HWeog], axis = 2)
    else:
        X = HWeeg

    # bidirectional frame-level recurrent layer
    with tf.device('/gpu:0'), tf.variable_scope("frame_rnn_layer", reuse=reuse) as scope:
#        fw_cell1, bw_cell1 =bidirectional_recurrent_layer_bn_new(config.nhidden1,
#                                                                  config.nlayer1,
#                                                                  seq_len=config.frame_seq_len,
#                                                                  is_training=istraining,
#                                                                  input_keep_prob=dropout_keep_prob_rnn,
#                                                                  output_keep_prob=dropout_keep_prob_rnn) 
#        rnn_out1, rnn_state1 = bidirectional_recurrent_layer_output_new(fw_cell1,
#                                                                    bw_cell1,
#                                                                    X,
#                                                                    frame_seq_len,
#                                                                    scope=scope)
        fw_cell1, bw_cell1 = bidirectional_recurrent_layer(config.nhidden1,
                                                              config.nlayer1,
                                                              input_keep_prob=dropout_keep_prob_rnn,
                                                              output_keep_prob=dropout_keep_prob_rnn)
        rnn_out1, rnn_state1 = bidirectional_recurrent_layer_output(fw_cell1,
                                                                    bw_cell1,
                                                                    X,
                                                                    frame_seq_len,
                                                                    scope=scope)
        print(rnn_out1.get_shape())
        # output shape (batchsize*epoch_step, frame_step, nhidden1*2)

    with tf.device('/gpu:0'), tf.variable_scope("frame_attention_layer", reuse=reuse):
        features = attention(rnn_out1, config.attention_size1)
        print(features.get_shape())
        # attention_output1 of shape (batchsize, nhidden1*2)
        return features

