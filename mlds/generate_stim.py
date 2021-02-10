#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Generate all your stimuli 

Script that generates stimuli  for the MLDS experiment: 
perception of correlation in scaterplots. 

This is just a toy example, to show how the MLDS scripts work.

It contains two conditions, one for blue scatterplots and another for red
scatterplots. 

This is just an example; in the actual experiment the two or more conditions 
could be: type of distortion, which picture to distort, etc ...

The output of the script is all the images necessary for running the experiment.

Continues with --> generate_design.py


@author: G. Aguilar, June 2020
updated: Jan 2021
"""

import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# generate first set of scatterplots, blue ones. Condition BLUE
for i, r in enumerate([0.0, 0.25, 0.5, 0.75, 0.9, 0.98]):
    
    mean = [0, 0]
    cov = [[1, r], [r, 1]]  # diagonal covariance
    x = np.random.multivariate_normal(mean, cov, 1000)
    #r, p = pearsonr(x[:,0], x[:,1])
    
    plt.figure(figsize=(5,5))
    plt.plot(x[:,0], x[:,1], 'o')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    #plt.title('r = %f' % r)
    plt.savefig('r_%d_blue.png' % i)
    plt.close()


# generate second set of scaterplots, red ones. Conditon RED
for i, r in  enumerate([0.0, 0.25, 0.5, 0.75, 0.9, 0.98]):
    
    mean = [0, 0]
    cov = [[1, r], [r, 1]]  # diagonal covariance
    x = np.random.multivariate_normal(mean, cov, 1000)
    #r, p = pearsonr(x[:,0], x[:,1])
    
    plt.figure(figsize=(5,5))
    plt.plot(x[:,0], x[:,1], 'ro')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    #plt.title('r = %f' % r)
    plt.savefig('r_%d_red.png' % i)
    plt.close()

#  END
