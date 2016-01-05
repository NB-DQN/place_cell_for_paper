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

def hidden_to_coordinates(hidden):
    # open the model
    f = open('SVM_model.pkl', 'rb')
    clf =  pickle.load(f)
    f.close()
    
    # convert hidden layer to coordinates
    coordinates_pred_1d = clf.predict(hidden)