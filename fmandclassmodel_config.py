class Config(object):
    def __init__(self):
        # feature dimension
        self.ndim = 129
        # number of segments per subsequence
        self.frame_seq_len = 29
        # number of channels
        self.nchannel = 1
        self.nclass = 5

        self.learning_rate = 1e-4 # *32 E: adapted on 12/08. because I take the mean loss and not the sum as did Huy in his network
        self.l2_reg_lambda = 0.001 #/32
        self.training_epoch = 20
        self.batch_size = 32

        self.dropout_keep_prob_rnn = 0.75

        self.frame_step = self.frame_seq_len
        self.nhidden1 = 64
        self.nlayer1 = 1
        self.attention_size1 = 32
        #self.nhidden2 = 64
        #self.nlayer2 = 1
        self.mse_weight=0.5#0.01 #0.0016
        self.feature_extractor=True

        self.nfilter = 20
        self.nfft = 256
        self.samplerate = 100
        self.lowfreq = 0
        self.highfreq = 50

        self.evaluate_every = 500
        self.checkpoint_every = 500
