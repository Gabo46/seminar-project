#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs an MLDS experiment with the method of *quadruples* for perception of correlation
in scatterplots. This code runs an experiment replicating the example in 
Knoblauch & Maloney (2012).

python mlds_experiment_quad_correlation.py

Design and stimuli are dynamically generated.

Seminar: Image quality and human visual perception, WiSe 2020/21, TU Berlin

@author: G. Aguilar, June 2020, update Dec 2020

"""

import csv
import itertools
import random
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys, os

import pyglet
from pyglet import window
from pyglet import clock
from pyglet.window import key 


instructions = """
MLDS experiment with the method of quadruples\n
Press the UP or DOWN arrow
to indicate which pair is most different\n
Press ENTER to start
Press ESC to exit """


## stimulus presentation time variable
#presentation_time = 1 # presentation time in seconds, None for unlimited presentation
presentation_time = None


# vector with correlation values
r = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]


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


def create_design():
        
    trials = list(itertools.combinations(r, 4))
    #trials = list(itertools.combinations(range(len(r)), 3))

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
    
    new_data = {'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}

    return new_data

###############################################################################
class Experiment(window.Window):
    

    def __init__(self, *args, **kwargs):

        #Let all of the arguments pass through
        self.win = window.Window.__init__(self, *args, **kwargs)
        
        self.debug = False
        
        
        clock.schedule_interval(self.update, 1.0/30) # update at FPS of Hz
        
        # Setting up text objects
        self.welcome_text = pyglet.text.Label(instructions,
                                  font_name='Arial', multiline=True,
                                  font_size=25, x=int(self.width/2.0), y=int(self.height/2.0),
                                  width=int(self.width*0.75), color=(0, 0, 0, 255),
                                  anchor_x='center', anchor_y='center')
        
               
        # Results file - assigning filename
        global obsname
        rfls = glob.glob('%s_quads_*.csv' % obsname)
        
        if len(rfls)>0:
            nb = len(rfls)
        else:
            nb = 0
        
        self.resultsfile = '%s_quads_%d.csv' % (obsname, nb)
        
        # opening the results file, writing the header
        self.rf = open(self.resultsfile, 'w')
        self.resultswriter = csv.writer(self.rf)  
        header = ['resp', 'S1', 'S2', 'S3', 'S4']
        self.resultswriter.writerow(header)
    
        
        # experiment control 
        self.experimentphase = 0 # 0 for intro, 1 for running trials, 2 for good bye
        self.firstframe = True
        self.present_stim = True
        
        # calling some routines on start
        self.loaddesign()
        # forces a first draw of the screen
        self.dispatch_event('on_draw')

        
    def loaddesign(self):
        """ Loads the design file specifications"""
        #self.design = read_design_csv(self.designfile)
        self.design = create_design()
        self.totaltrials = len(self.design['S1'])
        
        if self.debug:
            print(self.design)
            print('total number of trials: %d ' % self.totaltrials)
            
        self.currenttrial = 0
    
    def update(self, dt):
        pass
    
    def on_draw(self):
        """ Executed when draws on the screen"""
        
        
        # clear the buffer
        pyglet.gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.clear()
        
        # ticks the clock
        #dt = clock.tick() # ticking the clock
        #print(f"FPS is {clock.get_fps()}")
        
        
        if self.debug:
            print('-------- ondraw')
            print('self.present_stim %d' % self.present_stim)
        
        
        if self.experimentphase == 0:
            if self.debug:
                print('experiment phase 0: welcome')
            # draws instruction text
            self.welcome_text.draw()
            
        # go through the trials 
        elif self.experimentphase == 1:
            
            if self.debug:
                print('experiment phase 1: going through the trials')
            
            
            # load images only on the first frame
            if self.firstframe:
                print('trial: %d' % self.currenttrial)
            
                # create and load images
                self.create_trial_images()
                self.load_images()
                
                # saves presentation time
                self.stimstarttime =  datetime.datetime.now()
            
            # draw images on the screen for a limited time
            if (presentation_time is None) or (self.present_stim):     
                self.image1.blit(int(self.width*0.25), int(self.height*0.75))
                self.image2.blit(int(self.width*0.75), int(self.height*0.75))
                self.image3.blit(int(self.width*0.25), int(self.height*0.25))
                self.image4.blit(int(self.width*0.75), int(self.height*0.25))
                   
            # timing
            self.firstframe = False
            
            # checking if stim time has passed
            if presentation_time is not None and self.present_stim:
                deltat = datetime.datetime.now() - self.stimstarttime
                
                if deltat.total_seconds() > presentation_time:
                    self.present_stim = False
            
        elif self.experimentphase == 2:
            if self.debug:
                print('experiment phase 2: goodbye')
            self.dispatch_event('on_close')  
            
        
        # flipping the buffers
        self.flip()
        #pyglet.gl.glFlush()
    
    def checkcontinue(self):
        """ Checks if we're at the end of the trials"""
        self.firstframe = True
        self.present_stim = True
        
        if self.currenttrial>=self.totaltrials:
             self.experimentphase = 2
             #self.dispatch_event('on_close')  
             
        
    def create_trial_images(self):
        
        r1 = self.design['S1'][self.currenttrial]
        r2 = self.design['S2'][self.currenttrial]
        r3 = self.design['S3'][self.currenttrial]
        r4 = self.design['S4'][self.currenttrial]
        
        self.create_image(r1, 's1.png')
        self.create_image(r2, 's2.png')
        self.create_image(r3, 's3.png')
        self.create_image(r4, 's4.png')
        
        return 0
        
    def create_image(self, r, imgname):
        
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
        plt.savefig(imgname)
        plt.close()
        
        return 0
        
        
        
    def load_images(self):
        """ Loads images of current trial """
        # load files 
        if self.debug:
            print('loading files')
            
        self.image1 = pyglet.image.load('s1.png')
        self.image2 = pyglet.image.load('s2.png')
        self.image3 = pyglet.image.load('s3.png')
        self.image4 = pyglet.image.load('s4.png')
        
        # changes anchor to the center of the image
        self.image1.anchor_x = self.image1.width // 2
        self.image1.anchor_y = self.image1.height // 2
        
        self.image2.anchor_x = self.image2.width // 2
        self.image2.anchor_y = self.image2.height // 2
        
        self.image3.anchor_x = self.image3.width // 2
        self.image3.anchor_y = self.image3.height // 2

        self.image4.anchor_x = self.image4.width // 2
        self.image4.anchor_y = self.image4.height // 2

    def savetrial(self, resp, resptime):
        """ Save the response of the current trial to the results file """
        
        s1 = self.design['S1'][self.currenttrial]
        s2 = self.design['S2'][self.currenttrial]
        s3 = self.design['S3'][self.currenttrial]
        s4 = self.design['S4'][self.currenttrial]
        
        row = [resp, r.index(s1)+1, r.index(s2)+1, r.index(s3)+1, r.index(s4)+1]
        # row to save is the indices in the vector R, plus one as MLDS package
        # likes indices to start at 1 and not zero.
        
        self.resultswriter.writerow(row)
        print('Trial %d saved' % self.currenttrial)
        
        
    ## Event handlers
    def on_close(self):
        """ Executed when program finishes """
        
        # if s1.png, s2.png and s3.png exists, erase.
        if os.path.exists("s1.png"):
            os.remove("s1.png")
        if os.path.exists("s2.png"):
            os.remove("s2.png")
        if os.path.exists("s3.png"):
            os.remove("s3.png")
        if os.path.exists("s4.png"):
            os.remove("s4.png")
  
    
        self.rf.close() # closing results csv file
        self.close() # closing window
        
    def on_key_press(self, symbol, modifiers):
        """ Executed when a key is pressed"""
        
        if symbol == key.ESCAPE:
            self.dispatch_event('on_close')  

        elif symbol == key.UP and self.experimentphase==1:
            print("Up")
            deltat = datetime.datetime.now() - self.stimstarttime
            self.savetrial(resp=0, resptime = deltat.total_seconds())
            self.currenttrial += 1
            self.checkcontinue()
            
        elif symbol == key.DOWN and self.experimentphase==1:
            print("Down")
            deltat = datetime.datetime.now() - self.stimstarttime
            self.savetrial(resp=1, resptime = deltat.total_seconds())
            self.currenttrial += 1
            self.checkcontinue()
            
        elif symbol == key.ENTER and self.experimentphase==0:
            if self.debug:
                print("ENTER")
            self.experimentphase += 1
        
        self.dispatch_event('on_draw') 
                 


#####################################################################
if __name__ == "__main__":
        
    # ask for observer name
    obsname  = input ('Please input the observer name (e.g. demo): ')

    # for fullscreen, use fullscreen=True and give your correct screen resolution in width= and height=
    win = Experiment(caption="MLDS experiment with the method of quadruples", 
                     vsync=False, height=800, width=1200, fullscreen=False)
    pyglet.app.run()


