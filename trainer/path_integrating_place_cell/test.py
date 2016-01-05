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

# set parameters
n_epoch = 50000 # 1000000 # number of epochs
n_units = 25 # number of units per layer, len(train)=5 -> 20 might be the best
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 50 # 1000 # epoch on which accuracy and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size_x = 9
maze_size_y = 9
offset_timing = 2

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

# generate dataset
"""
input data: direction (2D)
output target: coordinate (converted to 1D)
"""
def generate_seq(seq_length, maze_size_x, maze_size_y):
    directions = []
    locations_1d = [] # 1D coorinate

    locations_1d.append(0)

    current = (0, 0) # 2D coordinate
    for i in range(0, seq_length):
        direction_choice = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        if current[0] == maze_size_x-1:
            direction_choice.remove([1, 0, 0, 0])
        if current[0] == 0:
            direction_choice.remove([0, 1, 0, 0])
        if current[1] == maze_size_y-1:
            direction_choice.remove([0, 0, 1, 0])
        if current[1] == 0:
            direction_choice.remove([0, 0, 0, 1])
        direction = random.choice(direction_choice)

        if   direction == [1, 0, 0, 0]:
            current = (current[0] + 1, current[1]    )
        elif direction == [0, 1, 0, 0]:
            current = (current[0] - 1, current[1]    )
        elif direction == [0, 0, 1, 0]:
            current = (current[0]    , current[1] + 1)
        elif direction == [0, 0, 0, 1]:
            current = (current[0]    , current[1] - 1)

        directions.append(direction)
        locations_1d.append(current[0]+current[1]*maze_size_x)

    # directions = np.array(directions, dtype='float32')
    # locations_1d = np.array(locations_1d, dtype='int32')
    return directions, locations_1d

# test dataset
test_data, test_targets = generate_seq(100, maze_size_x, maze_size_y)

# model
f = open('pi.pkl', 'rb')
model =  pickle.load(f)
f.close()

# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())


# one-step forward propagation
def forward_one_step(data, targets, state, train=True):
    if args.gpu >= 0:
        data = cuda.to_gpu(data)
        targets = cuda.to_gpu(targets)
    x = chainer.Variable(data, volatile=not train)
    t = chainer.Variable(targets, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}
    return state, F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# initialize hidden state
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}

# evaluation


def evaluate(data, targets, test=False):
    sum_accuracy = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)

    for i in six.moves.range(len(data)):
        one_hot_target = inilist = [0] * 81
        if targets[i] % offset_timing == 0:
            #if  targets[i] % 2 == 0 and targets[i] // 9 % 2 == 0:
            one_hot_target[targets[i]] = 1
        x_batch = mod.array([data[i] + one_hot_target], dtype = 'float32')
        t_batch = mod.array([targets[i + 1]], dtype = 'int32')
        state, loss, accuracy = forward_one_step(x_batch, t_batch, state, train=False)
        sum_accuracy += accuracy.data
        if test == True:
            print('{} Target: ({}, {})'.format(accuracy.data, t_batch[0] % maze_size_x,
                                               t_batch[0] // maze_size_x))

    return int(cuda.to_cpu(sum_accuracy))

# Evaluate on test dataset
print('[test]')
test_perp = evaluate(test_data, test_targets, test=True)
print('test classified: {}/{}'.format(test_perp, len(test_data)))
