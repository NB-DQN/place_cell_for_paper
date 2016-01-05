"""
Pre-training the place cell

Model
LSTM with one hidden layer
I don't know if truncated BPTT or gradient clip are necessary here
"""

import argparse
import math
import sys
import time
import random
import pickle

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from chainer import computational_graph as c

from dataset_generator import DatasetGenerator

import matplotlib.pyplot as plt

# set parameters
n_epoch = 100000 # number of epochs

n_units = 60 # number of units per layer, len(train)=5 -> 20 might be the best
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 100 # 1000 # epoch on which error and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size = (9, 9)

train_data_length = [20, 100]


# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# generate dataset
dg = DatasetGenerator(maze_size)

# test dataset
def generate_test_dataset():
    return dg.generate_seq(100)

# model
test_data = generate_test_dataset()
f = open('pretrained_model_'+str(n_units)+'.pkl', 'rb')
model =  pickle.load(f)
f.close()
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# one-step forward propagation
def forward_one_step(x, t, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
    x = chainer.Variable(x, volatile=not train)
    t = chainer.Variable(t, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    # plt.plot(range(len(h.data[0])), h.data[0])
    # plt.show()
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}

    sigmoid_y = 1 / (1 + np.exp(-y.data))
    bin_y = np.round((np.sign(sigmoid_y - 0.5) + 1) / 2)

    error = ((t.data - sigmoid_y) ** 2).sum()
    bin_y_error = ((t.data - bin_y) ** 2).sum()
    return state, F.sigmoid_cross_entropy(y, t), error, bin_y_error, h.data[0]

# initialize hidden state
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
        dtype=np.float32),
        volatile=not train)
        for name in ('c', 'h')}

# evaluation
def evaluate(data, test=False):
    sum_error = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)

    hh = []
    bin_y_error_sum = 0.0

    for i in range(len(data['input'])):
        x_batch = mod.asarray([data['input'][i]], dtype = 'float32')
        t_batch = mod.asarray([data['output'][i]], dtype = 'int32')
        state, loss, error, bin_y_error, h_raw = forward_one_step(x_batch, t_batch, state, train=False)
        
        hh.append(h_raw)
        bin_y_error_sum += bin_y_error
        
        sum_error += error
        if test == True:
            pass
            # print('{} Target: ({}, {})'.format(error, t_batch[0] % maze_size[0],
            #    t_batch[0] // maze_size[0]))

            # print('c: {}, h: {}'.format(state['c'].data, state['h'].data)) # show the hidden states
    return cuda.to_cpu(sum_error), hh, bin_y_error_sum

# Evaluate on test dataset
print('[test]')
test_data = generate_test_dataset()
test_perp, test_hh, test_error = evaluate(test_data, test=True)
print('test error: {}'.format(test_error))
