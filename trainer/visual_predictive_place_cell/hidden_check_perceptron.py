# linkage of hidden outputs (test_hh) and coorinates (test_data['coordinates'])
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

ev_iterations = 10

test_data_stack = []
test_hh_stack = []
for i in range(ev_iterations):
    import test
    
    # Evaluate on test dataset
    test_data = test.generate_test_dataset()
    print('[test]')
    test_perp, test_hh, test_y_bin_error_sum = test.evaluate(test_data, test=True)
    print('test classified: {}'.format(test_perp))
    
    test_data_stack.extend(test_data['coordinates'])
    test_hh_stack.extend(test_hh)
   
N_train = ev_iterations * 100 * 4 / 5
N_test = ev_iterations * 100 / 5

input_data = test_hh_stack
output_data = []
for i in range(len(test_data_stack)):
    output_data.append(test_data_stack[i][0] + test_data_stack[i][1] * 9)

print(len(input_data))
print(len(output_data))

# data allocation
train_data, test_data = np.split(input_data,   [N_train])
train_targets, test_targets = np.split(np.array(output_data),   [N_train])

print(test_data)
print(test_targets)

# model
model = chainer.FunctionSet(l1=F.Linear(81, 81))

# forward propagation
def forward(x_data, t_data, train=True):
    # Neural net architecture
    x = chainer.Variable(x_data)
    t = chainer.Variable(t_data)
    y = F.relu(model.l1(x))
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
optimizer = optimizers.SGD(lr=0.1)
optimizer.setup(model.collect_parameters())

# number of epochs
n_epoch  =1000

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_train):
        x_batch = np.array(train_data, dtype = 'float32')
        t_batch = np.array(train_targets, dtype = 'int32')

        optimizer.zero_grads()
        loss, acc = forward(x_batch, t_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(t_batch)
        sum_accuracy += float(acc.data)
    
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N_train, sum_accuracy / N_train))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test):
        x_batch = np.array(test_data, dtype = 'float32')
        t_batch = np.array(test_targets, dtype = 'int32')

        loss, acc = forward(x_batch, t_batch, train=False)

        sum_loss += float(loss.data) * len(t_batch)
        sum_accuracy += float(acc.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
        
        