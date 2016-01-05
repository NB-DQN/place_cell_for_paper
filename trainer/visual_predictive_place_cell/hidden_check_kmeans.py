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

from sklearn.cluster import KMeans
from sklearn import metrics

ev_iterations = 100

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
classifier = KMeans(n_clusters=81) 
classifier.fit(X_train)
y_train_pred = classifier.predict(X_train) 

# show the train accuracy
labels_list = zip(y_train, y_train_pred)
labels_list.sort()
print(labels_list)


print(metrics.adjusted_rand_score(y_train, y_train_pred))

# annotation label -> target
# def annotator(labels):
#     for data, target, label in zip(X_train, y_train, labels):
#         print(label, target)

# test
# test_labels = kmeans_model.predict(X_test)
# test_predicted = annotator(test_labels)

# calculate accuracy
# accuracy = 0
# for i in range(len(test_predicted)):
#     if test_predicted[i] == y_test[i]:
#         accuracy += 1

# accuracy /= len(test_predicted)

# print(accuracy)