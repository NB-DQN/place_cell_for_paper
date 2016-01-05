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


# stack means and SEs
means = np.zeros(4)
ses = np.zeros(4)

def plot_fig8_1():
    file_id_list = ['20151029093656', '20151029093830', '20151027021411', '20151027015426']
    
    for i in [3,2,1,0]:
        # load files
        f = open('plot_' + file_id_list[i] + '_valid_errors_mean.pkl', 'rb')
        valid_errors_mean =  pickle.load(f)
        f.close()
        
        f = open('plot_' + file_id_list[i] + '_valid_errors_se.pkl', 'rb')
        valid_errors_se =  pickle.load(f)
        f.close()
        
        means[i] = min(valid_errors_mean)
        ses[i] = valid_errors_se[np.argmin(valid_errors_mean)]
    
    return means, ses

def plot_fig8_2():
    # offset timing = 4
    f = open('plot_20151029093656_valid_errors_mean.pkl', 'rb')
    valid_errors_mean =  pickle.load(f)
    f.close()
    
    f = open('plot_20151029093656_valid_errors_se.pkl', 'rb')
    valid_errors_se =  pickle.load(f)
    f.close()
    
    means[0] = min(valid_errors_mean)
    ses[0] = valid_errors_se[np.argmin(valid_errors_mean)]
    
    # offset timing = 3
    f = open('plot_20151029093830_valid_errors_mean.pkl', 'rb')
    valid_errors_mean =  pickle.load(f)
    f.close()
    
    f = open('plot_20151029093830_valid_errors_se.pkl', 'rb')
    valid_errors_se =  pickle.load(f)
    f.close()
    
    means[1] = min(valid_errors_mean)
    ses[1] = valid_errors_se[np.argmin(valid_errors_mean)]
    
    # offset timing = 2
    f = open('plot_20151027021411_valid_errors_mean.pkl', 'rb')
    valid_errors_mean =  pickle.load(f)
    f.close()
    
    f = open('plot_20151027021411_valid_errors_se.pkl', 'rb')
    valid_errors_se =  pickle.load(f)
    f.close()
    
    means[2] = valid_errors_mean[-2]
    ses[2] = valid_errors_se[-2]
    
    # offset timing = 1
    f = open('plot_20151027015426_valid_errors_mean.pkl', 'rb')
    valid_errors_mean =  pickle.load(f)
    f.close()
    
    f = open('plot_20151027015426_valid_errors_se.pkl', 'rb')
    valid_errors_se =  pickle.load(f)
    f.close()
    
    means[3] = valid_errors_mean[-1]
    ses[3] = valid_errors_se[-1]
    
    return means, ses

plot_fig8_1()

# plot
x = np.array([1.0/4, 1.0/3, 1.0/2, 1.0])
plt. errorbar(x, means, yerr = ses, fmt='ro-')
plt.title('LSTM errors')
plt.xlabel('Visual offset frequency')
plt.ylabel('error')
plt.ylim([-0.01, 0.10])

"""
plt.bar(np.array([0,1,2,3]), means, align = 'center', yerr = ses, ecolor = 'r')
plt.xticks(np.array([0,1,2,3]), ['1/4', '1/3', '1/2', '1/1'])
plt.title('LSTM errors')
plt.xlabel('Visual offset frequency')
plt.ylabel('error')
"""
"""
plt.bar(np.array([0,1,2,3]), 1-means, align = 'center', yerr = ses, ecolor = 'r')
plt.xticks(np.array([0,1,2,3]), ['1/4', '1/3', '1/2', '1/1'])
plt.title('LSTM accuracy')
plt.xlabel('Visual offset frequency')
plt.ylabel('accuracy')
plt.ylim([0.8, 1.01])
"""


d = datetime.datetime.today()

# save plots in PNG and SVG
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.svg')
plt.savefig('plot_' + d.strftime("%Y%m%d%H%M%S") + '.png')

# save x
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_means.pkl', 'wb')
pickle.dump(means, f, 2)
f.close()

# save train errors
f = open('plot_' + d.strftime("%Y%m%d%H%M%S") + '_ses.pkl', 'wb')
pickle.dump(ses, f, 2)
f.close()


plt.show()

