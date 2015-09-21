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

ev_iterations = 100

test_data_stack = []
test_hh_stack = []
for i in range(ev_iterations):
    import test
    
    # Evaluate on test dataset
    test_data = test.generate_test_dataset()
    test_perp, test_hh = test.evaluate(test_data, test=True)
    
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

# parameters
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['accuracy', 'precision', 'recall']
score = 'accuracy'

# for loop on scores

print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_estimator_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() / 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
print("Confusion martix")


print(confusion_matrix(y_true, y_pred))


# for fixed C, gamma and kernel
# C = 1.
#kernel = 'rbf'
# gamma  = 0.01

# estimator = SVC(C=C, kernel=kernel, gamma=gamma)
# classifier = OneVsRestClassifier(estimator)
# classifier.fit(train_x, train_y)
# pred_y = classifier.predict(test_x)