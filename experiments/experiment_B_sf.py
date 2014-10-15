#!/usr/bin/env python
"""

Testing different spatial frequency bandwidths

(c) Laurent Perrinet - INT/CNRS

This is the basis for the following paper:

    Claudio Simoncini, Laurent U. Perrinet, Anna Montagnini, Pascal Mamassian, Guillaume S. Masson. More is not always better: dissociation between perception and action explained by adaptive gain control. Nature Neuroscience, 2012.
    http://invibe.net/LaurentPerrinet/Publications/Simoncini12


"""

import MotionClouds as mc
import numpy as np
name = 'Simoncini12'
# generates MPEG movies
vext = '.mpg'
# generates MATLAB mat files (uncomment to activate)
#vext = '.mat'
# just generates PNG of first frame
# vext = '.png'

display = False
DEBUG = False

# uncomment to preview movies
#ext, display = None, True

#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
#fx, fy, ft = mc.get_grids(256, 256, 256)
#fx, fy, ft = mc.get_grids(512, 512, 128)
color = mc.envelope_color(fx, fy, ft)

name_ = mc.figpath + name

# explore parameters
for B_sf in [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]:
    name_ = mc.figpath + name + '-B_sf' + str(B_sf).replace('.', '_')
    if mc.anim_exist(name_, vext=vext):
        z = color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf, B_theta=np.inf)
#         mc.visualize(z, name=name_ + '_envelope')
        im = mc.rectif(mc.random_cloud(z))
#         mc.cube(im, name=name_ + '_cube')
        mc.anim_save(im, name_, display=False, vext=vext)
#         mc.anim_save(im, name_, display=False, vext='.gif')


if DEBUG: # control enveloppe's shape

    z_low = mc.envelope_gabor(fx, fy, ft, B_sf=0.037, loggabor=False)
    z_high = mc.envelope_gabor(fx, fy, ft, B_sf=0.15, loggabor=False)

    import pylab, numpy
    pylab.clf()
    fig = pylab.figure(figsize=(12, 12))
    a1 = fig.add_subplot(111)
    a1.contour(numpy.fliplr(z_low[:mc.N_X/2, mc.N_Y/2, mc.N_frame/2:].T), [z_low.max()*.5], colors='red')
    a1.contour(numpy.fliplr(z_high[:mc.N_X/2, mc.N_Y/2, mc.N_frame/2:].T), [z_high.max()*.5], colors='blue')
    a1.set_xlabel('spatial frequency')
    a1.set_ylabel('temporal frequency')
    fig.savefig(mc.figpath + name + '_envelope_overlap.pdf')

if DEBUG:
    # checking for different frequencies
    for sf_0 in [0.1 , 0.2, 0.3, 0.8]:
        name_ = mc.figpath + name + '-sf_0' + str(sf_0).replace('.', '_')
        z = color * mc.envelope_gabor(fx, fy, ft, sf_0=sf_0)
        mc.anim_save(mc.rectif(mc.random_cloud(z)), name_, display=display, vext=vext)

    # explore different speeds than (V_X = 1, V_Y =0)
    for V_X in [1./4, 1./2 , 1. , 2.0]:
        name_ = mc.figpath + name + '-V_X' + str(V_X).replace('.', '_')
        z = color * mc.envelope_gabor(fx, fy, ft, V_X=V_X)
        mc.visualize(z, name=name_)
        mc.anim_save(mc.rectif(mc.random_cloud(z)), name_, display=display, vext=vext)

    for V_Y in [0.5 , 1.0 , 2.0]:
        name_ = mc.figpath + name + '-V_Y' + str(V_Y).replace('.', '_')
        z = color*mc.envelope_gabor(fx, fy, ft , V_Y=V_Y)
        mc.visualize(z, name=name_)
        mc.anim_save(mc.rectif(mc.random_cloud(z)), name_, display=display, vext=vext)


    # same stimulus but with different seeds
    for seed in [123456, 123457, 123458, 123459]:
        name_ = mc.figpath + name + '-seed' + str(seed)
        z = mc.rectif(mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft), seed=seed))
        mc.anim_save(z, name_, display=display, vext=vext)


    # checking for different frequencies
    for sf_0 in [0.1 , 0.2, 0.3, 0.8]:
        for B_sf in [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]:
            name_ = mc.figpath + name + '-sf_0' + str(sf_0).replace('.', '_')  + '-B_sf' + str(B_sf).replace('.', '_')
            z = color * mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf)
            mc.visualize(fx, fy, ft, z, name=name_)
            mc.anim_save(mc.rectif(mc.random_cloud(z)), name_, display=display, vext=vext)

