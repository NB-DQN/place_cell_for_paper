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
from sklearn.metrics import confusion_matrix

from sklearn.mixture import GMM
from sklearn import metrics

ev_iterations = 100

n_units = 81

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
    
N_train = ev_iterations * 100 * 4 / 5
N_test = ev_iterations * 100 / 5

input_data = test_hh_stack
output_data = []
for i in range(len(test_data_stack)):
    output_data.append(test_data_stack[i][0] + test_data_stack[i][1] * 9)

print('')

# data allocation
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data)

# clustering
classifier = GMM(n_components=81, n_iter=500, covariance_type='tied')
classifier.fit(X_train)
y_train_pred = classifier.predict(X_train) 

# show the train accuracy
labels_list = zip(y_train, y_train_pred)
labels_list.sort()
print(labels_list)
