# use larger learning rate for PTB


import numpy as np

from layers import *
from network_tools import *
from climin import RmsProp, Adam, GradientDescent
from text import loader

import time
import pickle
import os

import matplotlib.pyplot as plt

plt.ion()
plt.style.use('kosh')
plt.figure(figsize=(12, 7))
plt.clf()

np.random.seed(np.random.randint(1213))

experiment_name = 'lstm_dropout'

text_file = 'ptb.txt'

periods = [1, 2, 10, 100]
vocabulary_size = 49
states = 512				# per clock
output = 512				# for all clocks
dinput = vocabulary_size
doutputs = vocabulary_size

sequence_length = 30

batch_size = 20
learning_rate = 5e-3
niterations = 10000
momentum = 0.9

dropout = 0.1

forget_every = 100
gradient_clip = (-1.0, 1.0)

sample_every = 1000
save_every = 1000
plot_every = 100

full_recurrence = False
learn_state = False

anneal = True
dynamic_forgetting = False

dynamic_mom = False

logs = {}

data = loader('data/' + text_file, sequence_length, batch_size)


def dW(W):
	load_weights(model, W)	
	input, target = data.fetch()
	output = forward(model, input)
	backward(model, target)
	
	gradients = extract_grads(model)
	clipped_gradients = np.clip(gradients, gradient_clip[0], gradient_clip[1])

	loss = -1.0 * np.sum(target * np.log(output)) / (sequence_length * batch_size)
	gradient_norm = (gradients ** 2).sum() / gradients.size
	clipped_gradient_norm = (clipped_gradients ** 2).sum() / gradients.size
	
	logs['loss'].append(loss)
	logs['smooth_loss'].append(loss * 0.01 + logs['smooth_loss'][-1] * 0.99)
	logs['gradient_norm'].append(gradient_norm) 
	logs['clipped_gradient_norm'].append(clipped_gradient_norm) 
	
	return clipped_gradients


os.system('mkdir results/' + experiment_name)
path = 'results/' + experiment_name + '/'

logs['loss'] = []
logs['smooth_loss'] = [np.log(vocabulary_size)]
logs['gradient_norm'] = []
logs['clipped_gradient_norm'] = []

'''
model = [
			Dropout(dropout),
			CRNN_HSN(dinput, states, output, periods=periods, first_layer=True, sigma=0.1),
			Dropout(dropout),
			CRNN_HSN(output, states, output, periods=periods, sigma=0.1),
 			Linear(output, doutputs),
 			Softmax()
 		]
'''
model = [
			Dropout(dropout),
			LSTM(dinput, states, fbias=0.0, sigma=0.1),
			Dropout(dropout),
			LSTM(states, states, fbias=0.0, sigma=0.1),
 			Linear(states, doutputs),
 			Softmax()
 		]

W = extract_weights(model)

optimizer = Adam(W, dW, learning_rate, momentum=momentum)

config = 'experiment_name = ' + str(experiment_name) + '\n' + 'periods = ' + str(periods) + '\n' + 'vocabulary_size = ' + str(vocabulary_size) + '\n' + 'states = ' + str(states) + '\n' + 'output = ' + str(output) + '\n' + 'dinput = ' + str(dinput) + '\n' + 'doutputs = ' + str(doutputs) + '\n' + 'sequence_length = ' + str(sequence_length) + '\n' + 'batch_size = ' + str(batch_size) + '\n' + 'learning_rate = ' + str(learning_rate) + '\n' + 'niterations = ' + str(niterations) + '\n' + 'momentum = ' + str(momentum) + '\n' + 'forget_every = ' + str(forget_every) + '\n' + 'gradient_clip = ' + str(gradient_clip) + '\n' + 'sample_every = ' + str(sample_every) + '\n' + 'save_every = ' + str(save_every) + '\n' + 'plot_every = ' + str(plot_every) + '\n' + 'full_recurrence = ' + str(full_recurrence) + '\n' + 'learn_state = ' + str(learn_state) + '\n' + 'anneal = ' + str(anneal) + '\n' + 'dynamic_forgetting = ' + str(dynamic_forgetting) + '\n' + 'text = ' + str(text_file) + '\n'
config += 'optimizer = ' + str(optimizer.__class__.__name__) + '\n' 
config += 'dropout = ' + str(dropout) + '\n' 

f = open(path + 'config.txt', 'w')
f.write(config)
f.close()

print config, 'Approx. Parameters: ', W.size

for i in optimizer:
	if i['n_iter'] > niterations:
		break

	print i['n_iter'], '\t',
	print logs['loss'][-1], '\t',
	print logs['gradient_norm'][-1]

	if i['n_iter'] % forget_every == 0:
		forget(model)

	if dynamic_mom:
		if i['n_iter'] > 100:
			optimizer.momentum = 0.9

	if dynamic_forgetting:
		if i['n_iter'] > 100:
			forget_every = 50
		if i['n_iter'] > 1000:
			forget_every = 500
		if i['n_iter'] > 10000:
			forget_every = 5000

	if i['n_iter'] % sample_every == 0:
		forget(model)
		x = np.zeros((20, vocabulary_size, 1))
		input, _ = data.fetch()
		x[:20, :, :] = input[:20, :, 0:1]
		ixes = []
		for t in xrange(1000):
			p = forward(model, np.array(x))
			p = p[-1]
			ix = np.random.choice(range(vocabulary_size), p=p.ravel())
			x = np.zeros((1, vocabulary_size, 1))
			x[0, ix, 0] = 1
			ixes.append(ix)
		sample = ''.join(data.decoder.to_c[ix] for ix in ixes)
		print '----' * 20
		print sample
		print '----' * 20
		forget(model)

	if anneal:
		if i['n_iter'] > 4000:
			optimizer.step_rate *= 0.1
		elif i['n_iter'] > 6000:
			optimizer.step_rate *=  0.1
	
	if i['n_iter'] % save_every == 0:
		print 'serializing model... '
		f = open(path + 'iter_' + str(i['n_iter']) +'.model', 'w')
		pickle.dump(model, f)
		f.close()

	if i['n_iter'] % plot_every == 0:
		plt.clf()
		plt.plot(logs['smooth_loss'])
		plt.draw()

print 'serializing logs... '
f = open(path + 'logs.logs', 'w')
pickle.dump(logs, f)
f.close()

print 'serializing final model... '
f = open(path + 'final.model', 'w')
pickle.dump(model, f)
f.close()

plt.savefig(path + 'loss_curve')
