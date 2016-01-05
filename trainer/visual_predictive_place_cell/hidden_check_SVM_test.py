# linkage of hidden outputs (test_hh) and coorinates (test_data['coordinates'])

import argparse
import math
import sys
import time
import random
import pickle

import numpy as np
import six

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

ev_iterations = 1000

n_units = 40

print(' ')
print('number of hidden units: ' + str(n_units))
print(' ')

test_data_stack = []
test_hh_stack = []
for i in range(ev_iterations):
    import test
    
    # Evaluate on test dataset
    test_data = test.generate_test_dataset()
    test_perp, test_hh, test_y_bin_error_sum = test.evaluate(test_data, test=True)
    
    test_data_stack.extend(test_data['coordinates'])
    test_hh_stack.extend(test_hh)

input_data = test_hh_stack
output_data = []
for i in range(len(test_data_stack)):
    output_data.append(test_data_stack[i][0] + test_data_stack[i][1] * 9)

print('')

# data allocation
# X_train, X_test, y_train, y_test = train_test_split(input_data, output_data)

# parameters
tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['accuracy', 'precision', 'recall']
score = 'accuracy'

print("# Tuning hyper-parameters for %s" % score)
print()

# SVM parameters
C = 1.
kernel = 'linear'
gamma  = 0.01

#load
f = open('SVM_model_' + str(n_units) + '.pkl', 'rb')
classifier =  pickle.load(f)
f.close()

pred = classifier.predict(input_data)
print(classification_report(output_data, pred))
