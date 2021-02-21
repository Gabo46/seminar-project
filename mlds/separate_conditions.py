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
# change the follwing variables as necessary
# list all the results files

image = 'cartoon'
observer = 'Samy'

files = [
    '../results/%s/%s/triads_0_results.csv' % (image, observer),
    '../results/%s/%s/triads_1_results.csv' % (image, observer),
    '../results/%s/%s/triads_2_results.csv' % (image, observer),
    '../results/%s/%s/triads_3_results.csv' % (image, observer),
    '../results/%s/%s/triads_4_results.csv' % (image, observer)
]
#files = ['design_GA_quadruples_0_results.csv', 'design_GA_quadruples_1_results.csv', 'design_GA_quadruples_2_results.csv']

# mapping: stimuli filenames --> stimulus vector, starting at one
# (vectors in R start at one, not like in python)
# Here you should put the order of the stimuli you chose
mappingA = {
    '../images/%s/original.jpg' % image: 1,
    '../images/%s/x2_scaled.jpg' % image: 2,
    '../images/%s/x4_scaled.jpg' % image: 3,
    '../images/%s/x8_scaled.jpg' % image: 4,
    '../images/%s/x16_scaled.jpg' % image: 5
}

mappingB = {
    '../images/%s/original.jpg' % image: 1,
    '../images/%s/30_scaled.jpg' % image: 2,
    '../images/%s/60_scaled.jpg' % image: 3,
    '../images/%s/75_scaled.jpg' % image: 4,
    '../images/%s/85_scaled.jpg' % image: 5,
    '../images/%s/95_scaled.jpg' % image: 6
}

# extend if you have more conditions
conditions = [mappingA, mappingB]  # extend this list if necessary
condnames = ['Factor', 'Compression']       # names of the conditons.
# the processed files will be saved using these suffixes

######################################################################
######################################################################

# iterate through files
for i, f in enumerate(files):

    print('** processing file: %s' % f)
    # opens file
    df = pd.read_csv(f)

    # iterate through conditions
    for c, mapping in enumerate(conditions):
        condname = condnames[c]

        # selects only the rows for the condition
        idx = (df['S3'].isin(list(mapping.keys()))) & (
            df['S1'].isin(list(mapping.keys())))
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
        fname = '../results/%s/%s/%d_%s.csv' % (image,
                                                observer, i, condname)
        print('saving as: %s' % fname)
        print('')

        slicedf.to_csv(fname, index=False)

        # ... to be read by R


# EOF
