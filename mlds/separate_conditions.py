#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Separate the conditons

The experiment was run with conditions intermixed randomly. Now we need to separate
the trials per condition, to later feed those files into R

Change the variables files, mappingA, B... conditions and condnames
as necessary (lines 24 - 48)

After running this script, you will have the CSV files ready to be analyzed with R

After separating, continues with the R script 
--> mlds_analysis.R


@author: 
"""

import pandas as pd

######################################################################
####### change the follwing variables as necessary
# list all the results files
files = [
    'designs/Gabriel/Gabriel_triads_0_results.csv',
    'designs/Gabriel/Gabriel_triads_1_results.csv',
    'designs/Gabriel/Gabriel_triads_2_results.csv',
    'designs/Gabriel/Gabriel_triads_3_results.csv',
    'designs/Gabriel/Gabriel_triads_4_results.csv'
]
#files = ['design_GA_quadruples_0_results.csv', 'design_GA_quadruples_1_results.csv', 'design_GA_quadruples_2_results.csv']

# mapping: stimuli filenames --> stimulus vector, starting at one 
# (vectors in R start at one, not like in python)
# Here you should put the order of the stimuli you chose
mappingA = {
    '../images/landscape/original.jpg': 1,
    '../images/landscape/x2_scaled.jpg': 2,
    '../images/landscape/x4_scaled.jpg': 3,
    '../images/landscape/x8_scaled.jpg': 4,
    '../images/landscape/x16_scaled.jpg': 5
    }

mappingB = {
    '../images/landscape/original.jpg' : 1,
    '../images/landscape/30_scaled.jpg': 2,
    '../images/landscape/60_scaled.jpg': 3,
    '../images/landscape/75_scaled.jpg': 4,
    '../images/landscape/85_scaled.jpg': 5,
    '../images/landscape/95_scaled.jpg': 6
    }

## extend if you have more conditions 
conditions = [mappingA, mappingB] # extend this list if necessary
condnames = ['Factor', 'Compression']       # names of the conditons. 
# the processed files will be saved using these suffixes

######################################################################
######################################################################

# iterate through files
for f in files:
    
    print('** processing file: %s' % f)
    # opens file
    df = pd.read_csv(f)
    
    # iterate through conditions
    for c, mapping in enumerate(conditions):
        condname = condnames[c]
        
        # selects only the rows for the condition
        idx = (df['S3'].isin(list(mapping.keys()))) | (df['S1'].isin(list(mapping.keys())))
        slicedf = df[idx].copy()
        
        print('condition %s has %d trials' % (condname, len(slicedf)))
        
        
        # applies the mapping
        slicedf['S1'] = slicedf['S1'].map(mapping)
        slicedf['S2'] = slicedf['S2'].map(mapping)
        slicedf['S3'] = slicedf['S3'].map(mapping)
        try:
            slicedf['S4'] = slicedf['S4'].map(mapping)
            
        except:
            print('this is a triad experiment')
            
        # finally we save the newly coded file,
        fname = '%s_%s.csv' % (f.split('.')[0], condname)
        print('saving as: %s' % fname)
        print('')

        slicedf.to_csv(fname, index=False)
        
        # ... to be read by R
        
    
# EOF
    
