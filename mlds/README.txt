### Example how to organize the code to run an MLDS experiment

## The workflow is the following:

1. Generate the stimuli: generate_stim.py  
The script creates two sets of scatterplot stimuli in two  "conditions", 
a "red" condition with red dots, and a "blue" condition with blue dots. 
This is just a toy example, in reality you could have more conditions 
as type of distortion, or images, or both.


2. Generate the design files :  generate_design.py

Creates the design files. The design files are the files that tell
the experiment script which images to show first, which second, etc.

The design file contains all the triads or quadruples together for all
conditions, in random order.

3. Run the experiment:  mlds_experiment_triads.py  or mlds_experiment_quadruples.py

The script that actually runs the experiment. It reads the design file,
shows the images and saves the responses. The design file + responses 
are save in a 'results' file.

4. Preprocessing data: separate_conditions.py

Because the conditons were presented intermixed, we need to separate them
before analysing the data in R. 
Here the script separates the two conditons and save new CSV files with
the separated data.
And it also changes the labelling of the files. Instead of the images filanames,
it maps to the stimulus vector that the MLDS package in R "understands".


5. MLDS analysis in R:  mlds_analysis.R

Loads the separated file belonging to the same condition, and fits the scale
(as you've seen in a previous session)



# END  # GA Jan 2021

