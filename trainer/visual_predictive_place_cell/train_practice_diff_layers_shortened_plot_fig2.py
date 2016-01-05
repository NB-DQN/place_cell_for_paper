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
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from dataset_generator import DatasetGenerator

# set parameters
n_epoch = 10000 # number of epochs
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
        test_mean_squared_error, test_hh = evaluate(test_data, True)
        if test == True:
            label.append(test_data['coordinates'])
            input_data. append(test_hh)
        else:
            label.extend(test_data['coordinates'])
            input_data.extend(test_hh)
        
    return input_data , label

# grid search function using defalut module
def grid_search_1(train_X, train_Y):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_Y)
    return clf

# grid search function using scratched function
def grid_search_2(train_X, train_Y):
    C = np.logspace(-4, 4, 10)
    Gamma = np.logspace(-4, 4, 10)
    max_score = 0.0
    for g in Gamma:
        print(g)
        row = list()
        for c in C:
            estimator = SVC(C=c, kernel='linear', gamma=g)
            classifier = OneVsRestClassifier(estimator)
            classifier.fit(train_X, train_Y)
            pred_train = classifier.predict(train_X)
            score = accuracy_score(train_Y, pred_train) 
            row.append(score)
            if max_score < score:
                max_score = score
                max_classifier = classifier 
    return max_classifier

# SVM with fixed hyper parameters
def svm_fixed(train_X, train_Y):
    C = 1.
    kernel = 'linear'
    gamma  = 0.01
    estimator = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(train_X, train_Y)
    return classifier

# stack results
lstm_errors_mean = np.zeros(len(list_n_units))
lstm_errors_se = np.zeros(len(list_n_units))
svm_errors_mean = np.zeros(len(list_n_units))
svm_errors_se = np.zeros(len(list_n_units))


# loop initialization
for j in range(len(list_n_units)):
    n_units = list_n_units[j]
    print('n_units = {}'.format(n_units))
    
    f = open('pretrained_model_'+str(n_units)+'.pkl', 'rb')
    model =  pickle.load(f)
    f.close()
    
    print('calculate LSTM errors')
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
    print('start SVM')
    # generate dataset for SVM
    svm_X_train, svm_y_train = generate_seq_sklearn(ev_iterations, False)
    svm_X_test, svm_y_test = generate_seq_sklearn(ev_iterations / 5, True)
    
    # SVM
    clf = svm_fixed(svm_X_train, svm_y_train)
    # clf = grid_search_1(svm_X_train, svm_y_train)
    # clf = grid_search_2(svm_X_train, svm_y_train)
    
    print('SVM test')
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

plt. errorbar(x, lstm_errors_mean, yerr = lstm_errors_se, fmt='bo-')
plt.hold(True)
plt. errorbar(x, svm_errors_mean, yerr = svm_errors_se, fmt='go-')
plt.title('LSTM and SVM error versus hidden size')
plt.xlabel('size of hidden units')
plt.ylabel('errors')
plt.xlim([15, 65])
plt.legend(['LSTM', 'SVM'], loc =1)

"""
fig, ax1 = plt.subplots()
ax1. errorbar(x, lstm_errors_mean, yerr = lstm_errors_se, fmt='bo-')
ax2 = ax1.twinx()
ax2. errorbar(x, svm_errors_mean, yerr = svm_errors_se, fmt='go-')
ax1.set_title('LSTM and SVM error versus hidden size')
ax1.set_xlabel('size of hidden units')
ax1.set_xlim([15, 65])
ax1.set_ylabel('LSTM error', color='b')
ax2.set_ylabel('SVM error', color='g')
"""
d = datetime.datetime.today()

# save plots in PNG and SVG
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.svg')
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.png')

# save variables
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_x.pkl', 'wb')
pickle.dump(x, f, 2)
f.close()

f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_lstm_errors_mean.pkl', 'wb')
pickle.dump(lstm_errors_mean, f, 2)
f.close()

f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_lstm_errors_se.pkl', 'wb')
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

