#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Generate a design file

Script that writes a design file for an MLDS experiment with triads and
with quadruples.

It includes two conditions (here condition RED and BLUE, just as an example).

Change the variables Nrep, obsname, condA, condB and exp as necessary
(lines 104 - 111)
And add 

Continues with --> mlds_experiment_triads.py or
                   mlds_experiment_quadruples.py


@author: G. Aguilar, June 2020
updated: Jan 2021

"""

import itertools
import random
import numpy as np
import pandas as pd
import random
import os

##############################################################################
def shuffle_triad(t):
    
    # randomly chooses if triad is increasing or decreasing
    if random.randint(0,1)==1:
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
    else:
        t1 = t[2]
        t2 = t[1]
        t3 = t[0]
        
    return (t1, t2, t3)


def shuffle_quadruple(t):
    
    # randomly chooses if triad is increasing or decreasing
    if random.randint(0,1)==1:
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
        t4 = t[3]
    else:
        t1 = t[3]
        t2 = t[2]
        t3 = t[1]
        t4 = t[0]
        
    return (t1, t2, t3, t4)


def create_design_quadruples(v):
        
    trials = list(itertools.combinations(v, 4))
    
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for t in trials:
        t1, t2, t3, t4 = shuffle_quadruple(t)
        
        s1.append(t1)
        s2.append(t2)
        s3.append(t3)
        s4.append(t4)
    
    data = {'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}

    return data


def create_design_triads(v):
        
    trials = list(itertools.combinations(v, 3))
    
    s1 = []
    s2 = []
    s3 = []
    for t in trials:
        t1, t2, t3 = shuffle_triad(t)
        
        s1.append(t1)
        s2.append(t2)
        s3.append(t3)
        
    data = {'S1': s1, 'S2': s2, 'S3': s3}

    return data


##############################################################################
###################### Generting Design Files ################################
Nrep = 5         # number of repetitions. At least 3, 5 recommended 
obsname = 'Gabriel' # observer name, in my case, GA.

# vector with all stimuli filenames in each condition. In this example we have 
# two conditions, 'red' and 'blue', and 6 stimuli per condition
condA = ['../images/landscape/original/original.jpg', '../images/landscape/original/x2.jpg', '../images/landscape/original/x4.jpg', '../images/landscape/original/x8.jpg']

conditions = [condA] # extend this list if necessary

exp = 'quadruples' # 'triads' or 'quadruples'

### Added line: Create folder for observer if not exsists

outDir = './designs/%s' % obsname

if not os.path.exists(outDir):
    os.mkdir(outDir)

###############################################################################
for rep in range(Nrep):

    if exp=='triads':    
        trials = [create_design_triads(cond) for cond in conditions]
        
    elif exp=='quadruples':
        trials = [create_design_quadruples(cond) for cond in conditions]

    else:
        raise Exception('type of experiment not recognized')
    
    # creates pandas Dataframe for all trials in each condition
    dd = []
    for i, cond in enumerate(conditions):
        d = pd.DataFrame(trials[i])
        dd.append(d) 
    
    # joins the conditions
    df = pd.concat(dd)
        
    # shuffles order
    df.reset_index(drop=True, inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(drop=True, inplace=True)
    
    df.index.name = 'Trial'
    
    # save in design folder, under block number
    df.to_csv('designs/%s/%s_%s_%d.csv' % (obsname, obsname, exp, rep), index=False)
    

##############################################################################