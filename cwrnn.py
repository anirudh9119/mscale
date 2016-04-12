""" 
"""
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle
from collections import OrderedDict
from theano.ifelse import ifelse
import sys
import argparse
import random

logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class RNN(object):
    """    
    The first element of stride_size is the input size.
    The followed lement of stride_size is for the hidden layers.
    """
    def __init__(self, n_in, stride_size, n_out, activation=T.tanh, L1_reg=0.00, L2_reg=0.00):
        # Stride 0 is the input layer
        # SSPS stands for stride_size_partial_sum 
        SSPS = [0] * (len(stride_size) + 1)
        for idx in range(len(stride_size)):
            SSPS[idx+1] = SSPS[idx] + stride_size[idx]
        total_hidden_size = SSPS[-1]
        dummy = np.asarray(SSPS, dtype='int32')
        SSPS = theano.shared(value=dummy, name='SSPS') 

        def createRandomShare(size, name):
            dummy = np.asarray(np.random.uniform(size=size, low=-.01, high=.01), dtype=theano.config.floatX)
            share = theano.shared(value=dummy, name=name) 
            return share
        def createZeroShare(size, name):
            dummy = np.asarray(np.zeros(size), dtype=theano.config.floatX)
            upd = theano.shared(value=dummy, name=name) 
            return upd

        hidden_state = createZeroShare((total_hidden_size), 'hidden_state') 

        W_i2h = createRandomShare((n_in, total_hidden_size), 'W_h2h')
        W_i2h_updates = createZeroShare((n_in, total_hidden_size), 'W_h2h_updates')

        W_h2h = createRandomShare((total_hidden_size, total_hidden_size), 'W_h2h')
        W_h2h_updates = createZeroShare((total_hidden_size, total_hidden_size), 'W_h2h_updates')

        hidden_bias = createRandomShare((total_hidden_size), 'hidden_bias') 
        hidden_bias_updates = createZeroShare(total_hidden_size, 'hidden_state_bias') 

        W_h2o = createRandomShare((total_hidden_size, n_out), 'W_h2o')
        W_h2o_updates = createZeroShare((total_hidden_size, n_out), 'W_h2o_updates')

        output_bias = createRandomShare((n_out), 'output_bias') 
        output_bias_updates = createZeroShare((n_out), 'output_state_bias') 

        parameters = [W_i2h, W_h2h, hidden_bias, W_h2o, output_bias]
        updates = [W_i2h_updates, W_h2h_updates, hidden_bias_updates, W_h2o_updates, output_bias_updates]
        updates = OrderedDict(zip(parameters, updates))

        input = T.matrix()
        output = T.matrix()
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  

        # stride is count from 1
        def step(x_t, h_tm1):
            idx = SSPS[T.cast(x_t[0], 'int32')]
            i2h = W_i2h[:, :idx]
            h2h = W_h2h[:, :idx]
            bias = hidden_bias[:idx]
            hidden = activation(T.dot(x_t[1:], i2h) + T.dot(h_tm1, h2h) + bias)
            h_t = T.set_subtensor(h_tm1[:idx], hidden[:idx])
            y_t = T.dot(h_t, W_h2o) + output_bias
            return h_t, y_t

        def step1(x_t, h_tm1):
            h_t = activation(T.dot(x_t[1:], W_i2h) + T.dot(h_tm1, W_h2h) + hidden_bias)
            y_t = T.dot(h_t, W_h2o) + output_bias
            return h_t, y_t

        [new_hidden_state, predict], _ = theano.scan(step, sequences=input, outputs_info=[hidden_state, None])
        
        L1 = 0
        L2 = 0
        for param in parameters:
            L1 += abs(param.sum())
            L2 += (param ** 2).sum()

        cost = self._mse(output, predict) + L1_reg * L1 + L2_reg * L2
        gparams = []
        for param in parameters:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates_for_func = OrderedDict()
        for param, gparam in zip(parameters, gparams):
            weight_update = updates[param]
            upd = mom * weight_update - l_r * gparam
            updates_for_func[weight_update] = upd
            updates_for_func[param] = param + upd
        updates_for_func[hidden_state] = new_hidden_state[-1]

        self.error_func = theano.function(inputs=[input, output],
                                          outputs=self._mse(output, predict),
                                          mode=mode)

        self.train_func = theano.function(inputs=[input, output, l_r, mom],
                                          outputs=cost,
                                          updates=updates_for_func,
                                          mode=mode)

        self.debug_func = theano.function(inputs=[],
                                          outputs=[hidden_state],
                                          mode=mode)

        self.reset_hidden_func = theano.function(inputs=[], outputs = [], updates=[(hidden_state, T.zeros_like(hidden_state))])

    def train(self, x, y, l_r, mom):
        r = self.train_func(x, y, l_r, mom)
        return r

    def error(self, x, y):
        return self.error_func(x, y)

    def reset_hidden(self):
        self.reset_hidden_func()

    def debug(self, x, y):
        return self.debug_func()

    def _mse(self, y, ypred):
        return T.mean((ypred - y) ** 2)

class MetaRNN(BaseEstimator):
    def __init__(self, n_in=5, hidden_stride=[50], n_out=5, learning_rate=0.01,
                 L1_reg=0.00, L2_reg=0.00, learning_rate_decay=1,
                 activation='tanh', final_momentum=0.9, initial_momentum=0.5,
                 momentum_switchover=5):
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.activation = activation
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')
        self.rnn = RNN(n_in, hidden_stride, n_out, activation=activation, L1_reg = L1_reg, L2_reg = L2_reg)
        self.stride_cnt = len(hidden_stride)

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, nepochs=100, block_size=10):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (n_seq x n_in)
        Y_train : ndarray (n_seq x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        """

        if X_test is not None:
            assert(Y_test is not None)
            test_set_x, test_set_y = X_test, Y_test
            n_test  = test_set_x.shape[0]
        train_set_x, train_set_y = X_train, Y_train
        n_train = train_set_x.shape[0]

        def low_bit_order(n):
            k = 0
            while n % 2 ==0:
                n = n >> 1
                k += 1
            return k

        stride = np.array([low_bit_order(each % self.stride_cnt + 1)+1 for each in range(n_train)])
        stride = stride.reshape((n_train, 1))
        train_set_x = np.hstack((stride, train_set_x))

        if X_test is not None:
            stride = np.array([low_bit_order(each % self.stride_cnt + 1)+1 for each in range(n_test)])
            stride = stride.reshape((n_test, 1))
            test_set_x = np.hstack((stride, test_set_x))

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0

        sum_loss = 0;
        total_cnt = 1;
        module_size = 1;
        while (epoch < nepochs):
            epoch = epoch + 1
            self.rnn.reset_hidden()
            for idx in xrange(0, n_train, block_size):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum

                trainx = train_set_x[idx: idx + block_size]
                trainy = train_set_y[idx: idx + block_size]
                train_loss = self.rnn.train(trainx, trainy, self.learning_rate, effective_momentum)
                train_loss = np.mean(train_loss)
                sum_loss += train_loss

                if (total_cnt % module_size == 0):
                    module_size *= 2
                    logger.info('block: %s avg loss: %f, curr loss: %f, lr: %f' % (str(total_cnt).ljust(15), sum_loss / total_cnt, train_loss, self.learning_rate))
                total_cnt += 1

            if epoch % 10 == 0:
                self.rnn.reset_hidden()
                loss = [self.rnn.error(train_set_x[i: i+block_size], train_set_y[i: i+block_size]) for i in xrange(0, n_train, block_size)]
                train_loss = np.mean(loss)

                if X_test is not None:
                    self.rnn.reset_hidden()
                    loss = [self.rnn.error(test_set_x[i: i+block_size], test_set_y[i: i+block_size]) for i in xrange(0, n_test, block_size)]
                    test_loss = np.mean(loss)
                    logger.info('epoch %i, train loss: %f, test loss: %f, lr: %f' % (epoch, train_loss, test_loss, self.learning_rate))
                else:
                    logger.info('epoch %i, train loss: %f, lr: %f' % (epoch, train_loss, self.learning_rate))
            #self.learning_rate *= self.learning_rate_decay

def load_data(train_xpath, train_ypath, test_xpath = None, test_ypath = None):
    trainx = np.loadtxt(train_xpath)
    trainy = np.loadtxt(train_ypath)
    if (test_xpath != None):
        testx = np.loadtxt(test_xpath)
        testy = np.loadtxt(test_ypath)

    return trainx, trainy, testx, testy

def run(trainx, trainy, testx, testy, hidden_stride, epochs, block_size):
    t0 = time.time()

    n_in = trainx.shape[1]
    n_out = trainy.shape[1]
    model = MetaRNN(n_in=n_in, hidden_stride=hidden_stride, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    activation='tanh')
    model.fit(trainx, trainy, testx, testy, epochs, block_size)

    print "Elapsed time: %f" % (time.time() - t0)

def test_data(n_in, n_out, delay, n_seq):
    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_in)
    targets = np.zeros((n_seq, n_out))

    targets[delay:, :] = seq[:-delay, :] 
    targets += 0.01 * np.random.standard_normal(targets.shape)

    half = n_seq / 2
    trainx = seq[:half,]
    trainy = targets[:half,]
    testx = seq[half:,]
    testy = targets[half:,]

    return trainx, trainy, testx, testy


def test_data_zero_input(n_in, n_out, n_seq):
    np.random.seed(0)
    # simple lag test
    output = np.random.randn(n_seq, n_out)
    input = np.zeros((n_seq, n_in))

    return input, output, None, None 


def process_command_line(argv):
    """ Processing command line """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="""Version 0.1. cwrnn: Clockwise RNN. Execute this script using:
                                     OMP_NUM_THREADS=2 THEANO_FLAGS='device=cpu,openmp=True,exception_verbosity=high' python cwrnn.py""")

    # Manual:
    parser.add_argument("-trainx", dest="trainx", default=None, help="A data matrix of input for training")
    parser.add_argument("-trainy", dest="trainy", default=None, help="A data matrix of output for training")
    parser.add_argument("-testx", dest="testx", default=None, help="A data matrix of input for testing")
    parser.add_argument("-testy", dest="testy", default=None, help="A data matrix of output for testing")
    parser.add_argument("-s", dest="stride", required=True, help="Stride structure of the hidden stage. E.g. 10,5,5,3")
    parser.add_argument("-t", dest="test", help="Generate delayed test data and run the program. The format for parameter is n_in,n_out,delay,n_seq. E.g. 5,5,3,1000")
    parser.add_argument("-tz", dest="testz", help="Generate zero input test data and run the program. The format for parameter is n_in,n_out,delay,n_seq. E.g. 1,1,100.")
    parser.add_argument("-e", dest="nepochs", default=1000, help="The number of epochs to run")
    parser.add_argument("-b", dest="block_size", default=10, help="The number of lines for each block")

    return parser.parse_args(argv)

def main(argv=None):
    args = process_command_line(argv)
    if (args.test):
        params = [int(each) for each in args.test.split(',')]
        trainx, trainy, testx, testy = test_data(params[0], params[1], params[2], params[3])
    elif (args.testz):
        params = [int(each) for each in args.testz.split(',')]
        trainx, trainy, testx, testy = test_data_zero_input(params[0], params[1], params[2])
    else:
        trainx, trainy, testx, testy = load_data(args.trainx, args.trainy, args.testx, args.testy)

    stride = [int(each) for each in args.stride.split(',')]
    run(trainx, trainy, testx, testy, stride, int(args.nepochs), int(args.block_size))
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = main()
    sys.exit(status)
