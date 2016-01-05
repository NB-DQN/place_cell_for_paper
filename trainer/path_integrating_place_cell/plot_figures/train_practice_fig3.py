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
import datetime

import numpy as np
import six
import matplotlib.pyplot as plt

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

from dataset_generator import DatasetGenerator

# set parameters
n_epoch = 100000 # number of epochs
n_units = 25 # number of units per layer, len(train)=5 -> 20 might be the best
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 50 # 1000 # epoch on which accuracy and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size = (9, 9)

train_data_length = [100]
offset_timing = 1

valid_iter = 20

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')                    
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# validation dataset
valid_data_stack = []
for i in range(valid_iter):
    valid_data = DatasetGenerator(maze_size).generate_seq(100, offset_timing)
    valid_data_stack.append(valid_data)

# test dataset
test_data = DatasetGenerator(maze_size).generate_seq(100, offset_timing)

# model
model = chainer.FunctionSet(
    x_to_h = F.Linear(64, n_units * 4),
    h_to_h = F.Linear(n_units, n_units * 4),
    h_to_y = F.Linear(n_units, maze_size[0] * maze_size[1]))
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    
# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())


# one-step forward propagation
def forward_one_step(data, targets, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
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
def evaluate(data, test=False):
    sum_accuracy = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    
    for i in six.moves.range(len(data['input'])):
        x_batch = mod.asarray([data['input'][i]], dtype = 'float32')
        t_batch = mod.asarray([data['output'][i]], dtype = 'int32')
        state, loss, accuracy = forward_one_step(x_batch, t_batch, state, train=False)
        sum_accuracy += accuracy.data
        
        error = 1 - sum_accuracy / len(data['input'])
    return error # return error, not accuracy!!

# learning loop iterations
epoch = 0
print('[train]')
print('going to train {} epochs'.format(n_epoch))

# stack errors
train_errors = np.zeros(n_epoch / valid_len + 1)
valid_errors_mean = np.zeros(n_epoch / valid_len + 1)
valid_errors_se = np.zeros(n_epoch / valid_len + 1)

for loop in range(len(train_data_length)):

    # loop initialization
    whole_len = train_data_length[loop] 
    jump = whole_len // batchsize # = whole len
    cur_log_perp = mod.zeros(())
    start_at = time.time()
    cur_at = start_at
    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    
    # loop starts
    while epoch <= n_epoch:
        
        # initialize hidden state to 0
        state = make_initial_state()
        
        # train dataset
        if loop == 0:
            train_data = DatasetGenerator(maze_size).generate_seq(whole_len, offset_timing)
        else:
            train_data = DatasetGenerator(maze_size).generate_seq_remote(whole_len, offset_timing)
        
        if epoch % valid_len == 0:
        
            # calculate accuracy, cumulative loss & throughput
            train_perp = evaluate(train_data) # error
            valid_perp_stack = np.zeros(valid_iter)
            for i in range(valid_iter):
                valid_perp = evaluate(valid_data_stack[i]) # error
                valid_perp_stack[i] = valid_perp
            valid_perp_mean = np.mean(valid_perp_stack, axis=0)
            valid_errors_mean[epoch / valid_len] = valid_perp_mean
            valid_perp_se = np.std(valid_perp_stack, axis=0) / np.sqrt(valid_iter)
            valid_errors_se[epoch / valid_len] = valid_perp_se
            train_errors[epoch / valid_len] = train_perp
            
            if epoch == 0:
                perp = None
            else:
                perp = cuda.to_cpu(cur_log_perp) / valid_len
                perp = int(perp * 100) / 100.0
                
            now = time.time()
            
            if epoch == 0:
                throughput = 0.0
            else:
                throughput = valid_len / (now - cur_at)

            print('epoch {}: train perp: {} train classified {}/{}, valid classified {}/100 ({:.2f} epochs/sec)'
            .format(epoch, perp, whole_len * (1 - train_perp), whole_len, 100 * (1 - valid_perp_mean), throughput))
            
            cur_at = now
            
            #  termination criteria
            # if perp < 0.001:
            #     break
            # else:
            #     cur_log_perp.fill(0)
            cur_log_perp.fill(0)
            
        for i in six.moves.range(jump):
        
            # forward propagation
            x_batch = mod.array([train_data['input'][i]],  dtype = 'float32')
            t_batch = mod.array([train_data['output'][i]], dtype = 'int32')
            
            state, loss_i, acc_i = forward_one_step(x_batch, t_batch, state)
            accum_loss += loss_i
            cur_log_perp += loss_i.data.reshape(())
            
            # truncated BPTT
            if (i + 1) % bprop_len == 0:  
                optimizer.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
                optimizer.clip_grads(grad_clip) # gradient clip
                optimizer.update()
            
            sys.stdout.flush()
            
        epoch += 1  
        
        # save the model    
        f = open('pretrained_model_'+str(maze_size[0])+'_'+str(maze_size[1])+'.pkl', 'wb')
        pickle.dump(model, f, 2)
        f.close()      

# plot
x = np.arange(0, n_epoch + 1, valid_len)
plt. plot(x, train_errors, 'bo-')
plt.hold(True)
plt. errorbar(x, valid_errors_mean, yerr = valid_errors_se, fmt='ro-')
plt.title('LSTM errors of path integrating place cells')
plt.xlabel('training epochs')
plt.ylabel('error')
plt.legend(['train', 'test'], loc =1)
# plt.ylim([0, 0.05])
d = datetime.datetime.today()

# save plots in PNG and SVG
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.svg')
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.png')

# save x
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_x.pkl', 'wb')
pickle.dump(x, f, 2)
f.close()

# save train errors
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_train_errors.pkl', 'wb')
pickle.dump(train_errors, f, 2)
f.close()

# save mean of valid errors
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_valid_errors_mean.pkl', 'wb')
pickle.dump(valid_errors_mean, f, 2)
f.close()

# save SE of valid erros
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_se.pkl', 'wb')
pickle.dump(valid_errors_se, f, 2)
f.close()

plt.show()

