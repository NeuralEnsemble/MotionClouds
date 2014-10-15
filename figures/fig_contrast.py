#!/usr/bin/env python
"""
Exploring the effect of changing contrast and the method used.

(c) Laurent Perrinet - INT/CNRS

"""

import pylab
import numpy as np
import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

import matplotlib.pyplot as plt
import Image
import math



name = 'contrast_methods-'
#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft)
ext = '.zip'
contrast = 0.25
B_sf = 0.3

for method in ['Michelson', 'energy']:
    z = color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf)
    name_ = mc.figpath + name + method + '-contrast-' + str(contrast).replace('.', '_') + '-B_sf-' + str(B_sf).replace('.','_')
    if mc.anim_exist(name_):
        im = np.ravel(mc.random_cloud(z))
        im_norm = mc.rectif(mc.random_cloud(z), contrast, method=method, verbose=True)

        plt.figure()
        plt.subplot(111)
        plt.title('Michelson normalised Histogram Ctr: ' + str(contrast))
        plt.ylabel('pixel counts')
        plt.xlabel('grayscale')
        bins = int((np.max(im_norm[:])-np.min(im_norm[:])) * 256)
        plt.xlim([0, 1])
        plt.hist(np.ravel(im_norm), bins=bins, normed=False, facecolor='blue', alpha=0.75)
        plt.savefig(name_)

def image_entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = np.sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -np.sum([p * math.log(p, 2) for p in samples_probability if p != 0])

#img = Image.open(mc.figpath + 'grating-B_sf0_8.png')
#print image_entropy(img)

# XXX: If we normalise the histogram then the entropy base on gray levels is going to be the almost the same. Review the idea of entropy between narrowband and broadband stimuli.
