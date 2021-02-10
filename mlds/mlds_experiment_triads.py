#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Run the experiment

Runs an MLDS experiment with the method of triads

It runs through a design file in csv format, which should contain the filenames of the
images to be displayed. 

From the command line pass the name of the design file as the first parameter
E.g.

python mlds_experiment_triads.py design_GA_triads_0.csv

It saves the responses from the observer in a results file.



After running the experiment,
Continues with --> separate_conditions.py


Seminar: Image quality and human visual perception, TU Berlin
@author: G. Aguilar, June 2020
v2: it allows unlimited or limited presentation time. Change the global variable     presentation_time
v3: updates on WiSe 20/21

"""

import csv
import  datetime
import sys
import pyglet
from pyglet import window
from pyglet import clock
from pyglet.window import key 

instructions = """
MLDS experiment with the method of triads\n
Press the LEFT or the RIGHT arrow
to indicate which pair is most different\n
Press ENTER to start
Press ESC to exit """


## stimulus presentation time variable
#presentation_time = 1 # presentation time in seconds, None for unlimited presentation
presentation_time = None


def read_design_csv(fname):
    """ Reads a CSV design file and returns it in a dictionary"""
    
    design = open(fname)
    header = design.readline().strip('\n').split(',')
    #print header
    data   = design.readlines()
    
    new_data = {}
    
    for k in header:
        new_data[k] = []
    for l in data:
        curr_line = l.strip().split(',')
        for j, k in enumerate(header):
            new_data[k].append(curr_line[j])
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
        
       
        # Design file
        global designfile
        self.designfile = designfile
        
        # Results file - assigning filename
        s = designfile.split('.')
        s[-1] = '_results.csv'
        self.resultsfile = ''.join(s)
        
        # opening the results file, writing the header
        self.rf = open(self.resultsfile, 'w')
        self.resultswriter = csv.writer(self.rf)  
        header = ['resp', 'S1', 'S2', 'S3']
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
        self.design = read_design_csv(self.designfile)
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
            
                # load images
                self.load_images()
                
                # saves presentation time
                self.stimstarttime =  datetime.datetime.now()
            
            # draw images on the screen for a limited time
            if (presentation_time is None) or (self.present_stim):
                self.image1.blit(int(self.width*0.25), int(self.height*0.75))
                self.image2.blit(int(self.width*0.5), int(self.height*0.25))
                self.image3.blit(int(self.width*0.75), int(self.height*0.75))
                        
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
             
        
    def load_images(self):
        """ Loads images of current trial """
        # load files 
        if self.debug:
            print('loading files')
            
        self.image1 = pyglet.image.load(self.design['S1'][self.currenttrial])
        self.image2 = pyglet.image.load(self.design['S2'][self.currenttrial])
        self.image3 = pyglet.image.load(self.design['S3'][self.currenttrial])
        
        # changes anchor to the center of the image
        self.image1.anchor_x = self.image1.width // 2
        self.image1.anchor_y = self.image1.height // 2
        
        self.image2.anchor_x = self.image2.width // 2
        self.image2.anchor_y = self.image2.height // 2
        
        self.image3.anchor_x = self.image3.width // 2
        self.image3.anchor_y = self.image3.height // 2

        

    def savetrial(self, resp, resptime):
        """ Save the response of the current trial to the results file """
        
        row = [resp, self.design['S1'][self.currenttrial], self.design['S2'][self.currenttrial],
               self.design['S3'][self.currenttrial]]
        self.resultswriter.writerow(row)
        print('Trial %d saved' % self.currenttrial)
        
        
    ## Event handlers
    def on_close(self):
        """ Executed when program finishes """
        
        self.rf.close() # closing results csv file
        self.close() # closing window
        
    def on_key_press(self, symbol, modifiers):
        """ Executed when a key is pressed"""
        
        if symbol == key.ESCAPE:
            self.dispatch_event('on_close')  

        elif symbol == key.LEFT and self.experimentphase==1:
            print("Left")
            deltat = datetime.datetime.now() - self.stimstarttime
            self.savetrial(resp=0, resptime = deltat.total_seconds())
            self.currenttrial += 1
            self.checkcontinue()
            
        elif symbol == key.RIGHT and self.experimentphase==1:
            print("Right")
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
    
    
    if len(sys.argv) > 1:
        designfile = sys.argv[1]
        
    # it no argument passed, uses default design file    
    else:
        designfile = 'design_triads.csv'

    
    # for fullscreen, use fullscreen=True and give your correct screen resolution in width= and height=
    win = Experiment(caption="MLDS experiment with triads", 
                     vsync=False, height=800, width=1200, fullscreen=False)
    pyglet.app.run()

