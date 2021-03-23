#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for image manipulation and image quality metrics

@author: G. Aguilar, April 2020
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


############### Image quality metrics ###############
def mse(x, y):
    """ Mean squared error MSE, as in Eq. 1.1
    """
    
    # converts the matrix into a vector
    x = x.flatten()
    y = y.flatten()
    
    # assert(condition is true), keep going
    # if assert (condition is false), returns AssertionError and halts execution
    assert(len(x)==len(y))
    
    # length of the vector x
    N = len(x)
    
    # MSE formula 
    mse = (1.0/N)*np.sum((x - y)**2)
    
    return mse



############### Image manipulation utilities ###############
def show_im(img, warn=True):
    """ Shows an 8-bit grayscale image
    """
    if warn:
        if img.min()<0:
            print("warning: image's minimum is out of bounds --> display/save will crop")
        if img.max()>255:
            print("warning: image's maximum is out of bounds --> display/save will crop")
    plt.figure(); 
    plt.imshow(img, cmap='gray', vmin=0, vmax=255); 
    plt.axis('off')
    

def show_2ims(im1, im2):
    """ Shows two 8-bit grayscale images side by side and computes MSE
    """
    show_im(np.hstack((im1, np.zeros((im1.shape[0], 5)), im2)))
    plt.title('MSE = %d' % mse(im1, im2))
    
def show_horizontal_cut(im1, im2):
    """ Shows an horizontal luminance profiles across the middle of 
    two 8-bit grayscale images
    """
    midrow = int(im1.shape[0]/2)
    plt.figure()
    plt.plot(im1[midrow,:], linewidth=1); 
    plt.plot(im2[midrow,:], linewidth=1, zorder =-1)
    plt.show()


def show_horizontal_cut1(im):
    """ Shows an horizontal luminance profiles across the middle of 
    one 8-bit grayscale image
    """
    midrow = int(im.shape[0]/2)
    plt.figure()
    plt.plot(im[midrow,:], linewidth=1); 
    plt.show()
    
def write_array_to_image(filename, arr):
    """
    Save a 2D numpy array as a grayscale image file.

    Parameters
    ----------
    filename : str
        full path to the file to be creaated.
    arr : np.ndarray
        2D numpy array
        The data to be stored in the image. Values will be cropped to [0,255].
    """
    if Image:
        imsize = arr.shape
        im = Image.new('L', (imsize[1], imsize[0]))
        im.putdata(arr.flatten())
        im.save(filename)

def im_stats(im):
    """ Shows min,max and range of a grayscale image 
    """
    print('min: %d' % im.min())
    print('max: %d' % im.max())
    print('range: %d' % im.ptp()) # "ptp: point-to-point


#################################################################
### numerical utils    
def normalize_to_range(x, endrange = (0,1)):
    """ Function that normalizes an input array or matrix to a desired range 
    using a linear transformation 
    """
    
    a = (endrange[1] - endrange[0]) / (x.max() - x.min())  
    b = endrange[1] - a*x.max()
    
    return a*x+b 


# EOF
