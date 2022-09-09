import tensorflow as tf
import sys
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet')
#sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet')
from nn_basic_layers import *
from filterbank_shape import FilterbankShape
'''
Attentional recurrent neural network for sleep staging.
This version allows to build a different attention layer and epoch RNN layer. 
This code is based on the SeqSleepNet
https://github.com/pquochuy/SeqSleepNet  
(MIT © Huy Phan)

'''

def seqsleepnet_featureextractor_diffattn(config, input_x, dropout_keep_prob_rnn, frame_seq_len, epoch_seq_len, reuse=False, istraining=False, reuseattn=False, reuseepochrnn=False):


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
        Xeeg = tf.reshape(tf.squeeze(input_x[:,:,:,:,0]), [-1, config.ndim])
        # first filter bank layer
        Weeg = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
#        Weeg = tf.Variable(tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
        # non-negative constraints
        Weeg = tf.sigmoid(Weeg)
        # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
        Wfb = tf.multiply(Weeg,Wbl)
        HWeeg = tf.matmul(Xeeg, Wfb) # filtering
        HWeeg = tf.reshape(HWeeg, [-1, config.epoch_step, config.frame_step, config.nfilter])

    if(config.nchannel > 1):
        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eog", reuse=reuse):
            # Temporarily crush the feature_mat's dimensions
            Xeog = tf.reshape(tf.squeeze(input_x[:,:,:,:,1]), [-1, config.ndim])
            # first filter bank layer
            Weog = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
            # non-negative constraints
            Weog = tf.sigmoid(Weog)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb = tf.multiply(Weog,Wbl)
            HWeog = tf.matmul(Xeog, Wfb) # filtering
            HWeog = tf.reshape(HWeog, [-1, config.epoch_step, config.frame_step, config.nfilter])

    if(config.nchannel > 2):
        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-emg", reuse=reuse):
            # Temporarily crush the feature_mat's dimensions
            Xemg = tf.reshape(tf.squeeze(input_x[:,:,:,:,2]), [-1, config.ndim])
            # first filter bank layer
            Wemg = tf.get_variable('Variable', initializer=tf.random_normal([config.ndim, config.nfilter],dtype=tf.float32))
            # non-negative constraints
            Wemg = tf.sigmoid(Wemg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb = tf.multiply(Wemg,Wbl)
            HWemg = tf.matmul(Xemg, Wfb) # filtering
            HWemg = tf.reshape(HWemg, [-1, config.epoch_step, config.frame_step, config.nfilter])



    if(config.nchannel > 2):
        X = tf.concat([HWeeg, HWeog, HWemg], axis = 3)
    elif(config.nchannel > 1):
        X = tf.concat([HWeeg, HWeog], axis = 3)
    else:
        X = HWeeg

    X = tf.reshape(X, [-1, config.frame_step, config.nfilter*config.nchannel])


    # bidirectional frame-level recurrent layer
    with tf.device('/gpu:0'), tf.variable_scope("frame_rnn_layer", reuse=reuse) as scope:
        fw_cell1, bw_cell1 = bidirectional_recurrent_layer_bn_new(config.nhidden1,
                                                              config.nlayer1,
                                                              seq_len=config.frame_seq_len,
                                                              is_training=istraining,
                                                              input_keep_prob=dropout_keep_prob_rnn,
                                                              output_keep_prob=dropout_keep_prob_rnn)
        rnn_out1, rnn_state1 = bidirectional_recurrent_layer_output_new(fw_cell1,
                                                                        bw_cell1,
                                                                        X,
                                                                        frame_seq_len,
                                                                        scope=scope)
        print(rnn_out1.get_shape())
        # output shape (batchsize*epoch_step, frame_step, nhidden1*2) 64*2=128
    if not reuseattn:
        varscope="2/frame_attention_layer"        
    else:
        varscope="frame_attention_layer"
    with tf.device('/gpu:0'), tf.variable_scope(varscope, reuse=reuseattn):
        attention_out1 = attention(rnn_out1, config.attention_size1)
        print(attention_out1.get_shape())
        # attention_output1 of shape (batchsize*epoch_step, nhidden1*2)
    # if config.self_attention:
    #     attention_out2= self_attention(rnn_out1, config.attention_size1)
        
                                       
    e_rnn_input = tf.reshape(attention_out1, [-1, config.epoch_step, config.nhidden1*2])
    features=e_rnn_input
    if not reuseepochrnn:
        varscope="2/epoch_rnn_layer"        
    else:
        varscope="epoch_rnn_layer"
    # bidirectional frame-level recurrent layer
    with tf.device('/gpu:0'), tf.variable_scope(varscope, reuse=reuseepochrnn) as scope:
        fw_cell2, bw_cell2 = bidirectional_recurrent_layer_bn_new(config.nhidden2,
                                                              config.nlayer2,
                                                              seq_len=config.epoch_seq_len,
                                                              is_training=istraining,
                                                              input_keep_prob=dropout_keep_prob_rnn,
                                                              output_keep_prob=dropout_keep_prob_rnn)
        rnn_out2, rnn_state2 = bidirectional_recurrent_layer_output_new(fw_cell2,
                                                                        bw_cell2,
                                                                        e_rnn_input,
                                                                        epoch_seq_len,
                                                                        scope=scope)
        print(rnn_out2.get_shape())
        # output2 of shape (batchsize, epoch_step, nhidden2*2)

#        self.scores = []
#        self.predictions = []
#        with tf.device('/gpu:0'), tf.variable_scope("output_layer"):
#            for i in range(self.config.epoch_step):
#                score_i = fc(tf.squeeze(rnn_out2[:,i,:]),
#                                self.config.nhidden2 * 2,
#                                self.config.nclass,
#                                name="output-%s" % i,
#                                relu=False) #output: logits without softmax!
#                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
#                self.scores.append(score_i)
#                self.predictions.append(pred_i)
#


    return rnn_out2


def self_attention():
    with tf.device('/gpu:0'), tf.name_scope("self-attention"):
        tf.keras.layers.Conv1D()
    