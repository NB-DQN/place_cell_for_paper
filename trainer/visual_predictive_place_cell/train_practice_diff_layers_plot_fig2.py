"""
Pre-training the place cell

Model
lstm with one hidden layer
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

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from dataset_generator import DatasetGenerator

# set parameters
n_epoch = 15000 # number of epochs
# n_units = 60 # number of units per layer
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 25 # epoch on which accuracy and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size = (9, 9)

whole_len = 100 # seq length of training datasset
valid_iter = 20

ev_iterations = 100 # svm dataset

list_n_units = [20, 30, 40, 50, 60] # list_n_units = [60]

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# validation dataset
valid_data_stack = []
for i in range(valid_iter):
    valid_data = DatasetGenerator(maze_size).generate_seq(100)
    valid_data_stack.append(valid_data)

# test dataset
test_data = DatasetGenerator(maze_size).generate_seq(100)

# one-step forward propagation
def forward_one_step(x, t, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
    x = chainer.Variable(x, volatile=not train)
    t = chainer.Variable(t, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}
    
    sigmoid_y = 1 / (1 + np.exp(-y.data))
    mean_squared_error = ((t.data - sigmoid_y) ** 2).sum() / t.data.size

    return state, F.sigmoid_cross_entropy(y, t), mean_squared_error, h.data[0]

# initialize hidden state
def make_initial_state(batchsize=batchsize, train=True):
    global n_units
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
        dtype=np.float32),
        volatile=not train)
        for name in ('c', 'h')}

# evaluation
def evaluate(data, test=False):
    sum_error = 0
    state = make_initial_state(batchsize=1, train=False)
    hh = []
    
    for i in six.moves.range(len(data['input'])):
        x_batch = mod.asarray([data['input'][i]], dtype = 'float32')
        t_batch = mod.asarray([data['output'][i]], dtype = 'int32')
        state, loss, mean_squared_error, h_raw = forward_one_step(x_batch, t_batch, state, train=False)
        
        hh.append(h_raw)
        sum_error += mean_squared_error
    return sum_error / len(data['input']), hh

# generate dataset for svm
def generate_seq_sklearn(iterations, test):
    label = []
    input_data = []
    for i in range(iterations):
        test_data = DatasetGenerator(maze_size).generate_seq(100)
        test_mean_squared_error, test_hh = evaluate(test_data, test=True)
        if test == True:
            label.append(test_data['coordinates'])
            input_data. append(test_hh)
        else:
            label.extend(test_data['coordinates'])
            input_data.extend(test_hh)
        
    return input_data , label

# stack results
lstm_errors_mean = np.zeros(len(list_n_units))
lstm_errors_se = np.zeros(len(list_n_units))
svm_errors_mean = np.zeros(len(list_n_units))
svm_errors_se = np.zeros(len(list_n_units))

# for loop on different n_units
for j in range(len(list_n_units)):
    n_units = list_n_units[j]
    
    # model
    model = chainer.FunctionSet(
            x_to_h = F.Linear(64, n_units * 4),
            h_to_h = F.Linear(n_units, n_units * 4),
            h_to_y = F.Linear(n_units, 60))
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    
    # optimizer
    optimizer = optimizers.SGD(lr=1.)
    optimizer.setup(model.collect_parameters())
    
    # loop initialization
    jump = whole_len // batchsize # = whole len
    cur_log_perp = mod.zeros(())
    start_at = time.time()
    cur_at = start_at
    epoch = 0
    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    print('[train]')
    print('going to train {} epochs; n_units = {}'.format(n_epoch, n_units))
    
    # loop starts
    while epoch <= n_epoch:
    
        # initialize hidden state to 0
        state = make_initial_state()
    
        # train dataset
        train_data = DatasetGenerator(maze_size).generate_seq(whole_len)
    
        if epoch % valid_len == 0:
            
            # calculate accuracy, cumulative loss & throuput
            train_perp, hh = evaluate(train_data)
            valid_perp_stack = np.zeros(valid_iter)
            for i in range(valid_iter):
                valid_perp, hh = evaluate(valid_data_stack[i])
                valid_perp_stack[i] = valid_perp
            valid_perp_mean = np.mean(valid_perp_stack, axis=0)
            valid_perp_se = np.std(valid_perp_stack, axis=0) / np.sqrt(valid_iter)
            
            if epoch == 0:
                perp = None
            else:
                perp = cuda.to_cpu(cur_log_perp) / valid_len
                perp = int(perp * 100) / 100.0
                
            now = time.time()
            
            if epoch == 0:
                throuput =  0.0
            else:
                throuput = valid_len / (now - cur_at)
                
            print('epoch {}: train perp: {} train mean squared error: {:.5f}, valid mean squared error: {:.5f} ({:.2f} epochs/sec)'
                    .format(epoch, perp, train_perp, valid_perp_mean, throuput))
                    
            cur_at = now
            
            #  termination criteria
            if perp != None and perp < 2.5:
                break
            else:
                cur_log_perp.fill(0)
        
        for i in six.moves.range(jump):
    
            # forward propagation
            x_batch = mod.array([train_data['input'][i]],  dtype = 'float32')
            t_batch = mod.array([train_data['output'][i]], dtype = 'int32')
    
            state, loss_i, acc_i, h = forward_one_step(x_batch, t_batch, state)
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
        f = open('pretrained_model_'+str(n_units)+'.pkl', 'wb')
        pickle.dump(model, f, 2)
        f.close()
    
    # LSTM accuracy on validation dataset
    valid_perp_stack = np.zeros(valid_iter)
    for i in range(valid_iter):
        valid_perp, hh = evaluate(valid_data_stack[i])
        valid_perp_stack[i] = valid_perp
    valid_perp_mean = np.mean(valid_perp_stack, axis=0)
    valid_perp_se = np.std(valid_perp_stack, axis=0) / np.sqrt(valid_iter)
    lstm_errors_mean[j] = valid_perp_mean
    lstm_errors_se[j] = valid_perp_se
    
    # SVM
    # generate dataset for SVM
    svm_X_train, svm_y_train = generate_seq_sklearn(ev_iterations, False)
    svm_X_test, svm_y_test = generate_seq_sklearn(ev_iterations / 5, True)
    
    # SVM grid search parameters
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # SVM grid search
    print('start grid search')
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(svm_X_train, svm_y_train)
    
    # SVM test
    svm_perp_stack = np.zeros(ev_iterations / 5)
    for i in range(ev_iterations / 5):
        y_true, y_pred = svm_y_test[i], clf.predict(svm_X_test[i])
        svm_test_accuracy = accuracy_score(y_true, y_pred) # accuracy, not error
        svm_perp_stack[i] = svm_test_accuracy
    svm_perp_mean = np.mean(svm_perp_stack, axis=0)
    svm_perp_se = np.std(svm_perp_stack, axis=0) / np.sqrt(ev_iterations / 5)
    svm_errors_mean[j] = 1 - svm_perp_mean # error
    svm_errors_se[j] = svm_perp_se
    print('svm error:  {:.2f} '.format(svm_errors_mean[j]))

# plot
x = np.array(list_n_units)
"""
plt. errorbar(x, lstm_errors_mean, yerr = lstm_errors_se, fmt='bo-')
plt.hold(True)
plt. errorbar(x, svm_errors_mean, yerr = svm_errors_se, fmt='go-')
plt.title('LSTM and SVM error versus hidden size')
plt.xlabel('size of hidden units')
plt.ylabel('errors')
plt.legend(['LSTM', 'SVM'], loc =1)
"""

fig, ax1 = plt.subplots()
ax1. errorbar(x, lstm_errors_mean, yerr = lstm_errors_se, fmt='bo-')
ax2 = ax1.twinx()
ax2. errorbar(x, svm_errors_mean, yerr = svm_errors_se, fmt='go-')
ax1.title('LSTM and SVM error versus hidden size')
ax1.set_xlabel('size of hidden units')
ax1.set_ylabel('LSTM error', color='b')
ax2.set_ylabel('SVM error', color='g')

d = datetime.datetime.today()

# save plots in PNG and SVG
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.svg')
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.png')

# save variables
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_x.pkl', 'wb')
pickle.dump(x, f, 2)
f.close()

f = open('plot_ ' + d.strftime("%Y%m%d%H%M%S") + '_lstm_errors_mean.pkl', 'wb')
pickle.dump(lstm_errors_mean, f, 2)
f.close()

f = open('plot_ ' + d.strftime("%Y%m%d%H%M%S") + '_lstm_errors_se.pkl', 'wb')
pickle.dump(lstm_errors_se, f, 2)
f.close()

f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_svm_errors_mean.pkl', 'wb')
pickle.dump(svm_errors_mean, f, 2)
f.close()

# save SE of valid erros
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_svm_errors_se.pkl', 'wb')
pickle.dump(svm_errors_se, f, 2)
f.close()

plt.show()

