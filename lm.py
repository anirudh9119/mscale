'''
Build a simple neural language model using GRU units
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.extra_ops import  repeat
import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

profile = False
sys.setrecursionlimit(50000)



# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation, returns padded batch and mask
def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    # filter according to mexlen
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
 #   maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.

    return x, x_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, slow=False):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')


    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # Recurrent transformation weights for slower hiddens
    # to faster hiddens.
    if slow:
        W_h = norm_weight(dim, dim)
        params[_p(prefix, 'W_h')] = W_h
        #Should I include bias too ?
        #params[_p(prefix, 'b_h')] = numpy.zeros((dim,)).astype('float32')

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options,
              new_mask, prefix='gru',
              mask=None, one_step=False, init_state=None,
              slow_rnn = False, slow_rnn_states=None,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    state_hiddens=[]
    if slow_rnn:
        state_hiddens  = tensor.dot(slow_rnn_states, tparams[_p(prefix, 'W_h')])

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, new_mask_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h

        # if new_mask_ is 1, the first 2 terms, should be executed.
        # if new_mask_ is 0, the last term should be executed.

        h = new_mask_[:,None] * (m_[:, None] * h) + new_mask_[:,None]* ((1. - m_)[:, None] * h_) + ((1. - new_mask_)[:, None] * h_)
        return h


    # For slow RNN!
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice_slow(m_, x_, xx_, new_mask_, h_slow_ , h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_
        preactx = preactx + h_slow_

        # leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h

        # if new_mask_ is 1, the first 2 terms, should be executed.
        # if new_mask_ is 0, the last term should be executed.

        h = new_mask_[:,None] * (m_[:, None] * h) + new_mask_[:,None]* ((1. - m_)[:, None] * h_) + ((1. - new_mask_)[:, None] * h_)

        return h


    # prepare scan arguments
    seqs = []
    shared_vars = []
    _step = _step_slice
    seqs = [mask, state_below_, state_belowx, new_mask]
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]
    if slow_rnn:
        _step = _step_slice_slow
        seqs = [mask, state_below_, state_belowx, new_mask, state_hiddens]
    else:
        seqs = [mask, state_below_, state_belowx, new_mask]


    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:  # sampling
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:  # training
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval]
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              slow=True)


    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_slow',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              slow=False)

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params

def get_sliced_hiddens(slow_hidden, alpha):
    # num_seqs X batch_size X hidden_state_size
    # 30 x 32 x 1024

    Y = slow_hidden[alpha,:,:]
    return Y

def get_interpolated_hiddens(old_hidden,  n_timesteps, n_samples):
    '''
        number_of_reduced_timstamps
        alphas  = [1, 0.8, 0.6, 0.4, 0.2]
        T.sum(alpha)
    '''
    alpha = [0.5, 0.5]
    hidden_size = 1024
    batch_size = 32
    num_cons_hiddens = 14
    number_interp = 2

    #alpha = theano.vector('alpha')
    X  = old_hidden.dimshuffle(1, 0, 2)
    new_matrix2 = repeat(X, 2, axis=1)
    new_matrix2 = tensor.roll(new_matrix2, -1, axis=1)
    new_matrix2 = new_matrix2[:, 0:n_timesteps-2, :]
    new_matrix2 = new_matrix2.reshape([n_samples, (n_timesteps-2)/2, 2, hidden_size])

    def _step_slice(m_, alpha):
        return [alpha[0]* m_[0] + (1-alpha[0])*m_[1], alpha[1]* m_[0] + (1-alpha[1])*m_[1]]

    _step = _step_slice

    def step_batch(m_, alpha):
        seqs = m_
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    non_sequences=[alpha])
        return rval

    _batch_step = step_batch
    seqs = new_matrix2
    rval, updates = theano.scan(_batch_step,
                                sequences=seqs,
                                non_sequences=[alpha])
    out=[]
    out_batch =[]
    for batch_index in range(batch_size):
        for i in range(num_cons_hiddens):
            something =  [rval[j][batch_index][i] for j in range(number_interp)]
            if i==0:
                out = something
            if i >=1:
                out  = tensor.concatenate([out, something], axis=0)
        if batch_index == 0:
            out_batch = out
        if batch_index == 1:
            out_batch = tensor.stacklists([out_batch, out])
        if batch_index > 1:
            out = tensor.reshape(out,[1, n_timesteps-2, hidden_size])
            out_batch = tensor.concatenate([out_batch, out])

    zero_pad = tensor.zeros([out_batch.shape[0], number_interp , out_batch.shape[2]])
    out_batch = tensor.concatenate([zero_pad, out_batch], axis=1)



#    compute = theano.function([X, alpha],
#                              [out_batch, out],
#                              allow_input_downcast=True,
#                              on_unused_input='warn')
    return out_batch


# build a training model
def build_model(tparams, options, new_mask, alpha):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    opt_ret['emb'] = emb

    # pass through gru layer, recurrence here
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder_slow',
                                            mask=x_mask,
                                            new_mask=new_mask)

    sliced_hiddens = get_sliced_hiddens(proj[0], alpha)
    interp_hiddens = get_interpolated_hiddens(sliced_hiddens,
                                              n_timesteps,
                                              n_samples)

    interp_hiddens = interp_hiddens.dimshuffle(1,0,2)
    slow_rnn_hidden_states = interp_hiddens.astype('float32')

    slow_rnn_hidden_states =  proj[0]
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            new_mask=new_mask,
                                            slow_rnn = True,
                                            slow_rnn_states = slow_rnn_hidden_states)

    proj_h = proj[0]
    opt_ret['proj_h'] = proj_h



    #get_hidden = theano.function([x, x_mask], interp_hiddens)
    get_hidden = theano.function([x, x_mask], interp_hiddens.astype('float32'))

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    opt_ret['cost_per_sample'] = cost
    cost = (cost * x_mask).sum(0)

    return trng, use_noise, x, x_mask, opt_ret, cost, get_hidden


# build a sampler
def build_sampler(tparams, options, trng, new_mask):
    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')


    y_slow = tensor.vector('y_sampler', dtype='int64')
    init_state_slow = tensor.matrix('init_state', dtype='float32')


    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y])


    emb_slow = tensor.switch(y_slow[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y_slow])


    # apply one step of gru layer
    proj = get_layer(options['encoder'])[1](tparams, emb_slow, options,
                                            new_mask,
                                            prefix='encoder_slow',
                                            mask=None,
                                            one_step=True,
                                            init_state=init_state_slow)

    slow_rnn_state  = proj[0][0]

    # apply one step of gru layer
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            new_mask,
                                            prefix='encoder',
                                            mask=None,
                                            one_step=True,
                                            init_state=init_state,
                                            slow_rnn = True,
                                            slow_rnn_states=slow_rnn_state)


    next_state = proj[0][0]
    # compute the output probability dist and sample
    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, init_state,y_slow, init_state_slow]
    outs = [next_probs, next_sample, next_state, slow_rnn_state]


    inps_slow = [y_slow, init_state_slow]
    outs_slow = [slow_rnn_state]

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    f_next_slow = theano.function(inps_slow,
                                  outs_slow,
                                  name='f_next_slow',
                                  profile=profile)
    print 'Done'

    return f_next, f_next_slow


def get_interp_states(previous_state, new_state):

    m_ = [previous_state, new_state]
    alpha = [0.5, 0.5];
    return [alpha[0]* m_[0] + (1-alpha[0])*m_[1], alpha[1]* m_[0] + (1-alpha[1])*m_[1]]

# generate sample
#TODO:remove hard numbers!
def gen_sample(tparams, f_next, f_next_slow, options, trng=None, maxlen=30, argmax=False):

    sample = []
    sample_score = 0

    # initial token is indicated by a -1 and initial state is zero
    next_w = -1 * numpy.ones((1,)).astype('int64')
    next_state = numpy.zeros((1, options['dim'])).astype('float32')

    previous_state = next_state

    for ii in xrange(maxlen):
        next_state =  numpy.reshape(next_state, [1,1024])
        inps_slow = [next_w, next_state]
        ret_slow = f_next_slow(*inps_slow)

        interp_state = numpy.reshape(ret_slow[0], [1,1024])
        new_states = get_interp_states(previous_state, interp_state)
        previous_state = interp_state

        for index in range(2):

            temp =  numpy.reshape(new_states[index],[1,1024])
            #print temp.shape

            temp2 =  numpy.reshape(next_state, [1,1024])
            #print temp2.shape
            inps = [next_w, temp2,  next_w, temp]

            #print inps[0].shape
            #print inps[1].shape
            #print inps[2].shape
            #print inps[3].shape
            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break

    return sample, sample_score


# calculate the log probablities on a given corpus using language model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x in iterator:
        n_done += len(x)

        x, x_mask = prepare_data(x, n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):

    # allocate gradients and set them all to zero
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]

    # create gradient copying list,
    # from grads (tensor variable) to gshared (shared variable)
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    # define the update step rule
    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]

    # compile a function for update
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of GRU units
          encoder='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          n_words=100000,  # vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset='/data/lisatmp3/chokyun/wikipedia/extracted/wiki.tok.txt.gz',
          valid_dataset='../data/dev/newstest2011.en.tok',
          dictionary='/data/lisatmp3/chokyun/wikipedia/extracted/'
          'wiki.tok.txt.gz.pkl',
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)

    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    # create shared variables for parameters
    tparams = init_tparams(params)
    print tparams

    new_mask = tensor.alloc(1, maxlen)
    qw = tensor.extra_ops.cumsum(new_mask)
    qw2 = tensor.switch(tensor.eq(tensor.mod(qw,2), 0), 1, 0)
    temp = qw2
    alpha = qw2
    for i in range(31):
        qw2 = tensor.concatenate([qw2,temp], axis=0)

    qw2 = qw2.reshape([32,30])
    qw2 = qw2.T
    f=theano.function([new_mask], qw2)
    #print f(numpy.ones(maxlen).astype('int8'))
    new_mask =  qw2

    alpha = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]

    # build the symbolic computational graph

    trng, use_noise, \
        x, x_mask, \
        opt_ret, \
        cost, get_hidden = \
        build_model(tparams, model_options, new_mask, alpha)
    inps = [x, x_mask]

    print 'Buliding sampler'
    new_mask = tensor.alloc(1, maxlen)
    f_next, f_next_slow = build_sampler(tparams, model_options, trng, new_mask)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            # pad batch and create mask
            x, x_mask = prepare_data(x, maxlen=maxlen, n_words=n_words)
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()
            proj_h = get_hidden(x, x_mask)
            print numpy.asarray(proj_h).shape
            #ipdb.set_trace()
            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(5):
                    sample, score = gen_sample(tparams, f_next, f_next_slow,
                                               model_options, trng=trng,
                                               maxlen=30, argmax=False)
                    print 'Sample ', jj, ': ',
                    ss = sample
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r:
                            print worddicts_r[vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
