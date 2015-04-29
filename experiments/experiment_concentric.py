#!/usr/bin/env python
"""
Averaging over multiple clouds 

TODO: force the phase by setting the luminance at some point and averaging voer multiple instances

rm results/concentric.mp4; python experiment_concentric.py; open results/concentric.mp4 

(c) Laurent Perrinet - INT/CNRS

"""

import MotionClouds as mc
import numpy as np

name = 'concentric'
play = False #True
play = True

#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft)

name_ = mc.figpath + name

seed = 123456
im = np.zeros((mc.N_X, mc.N_Y, mc.N_frame))
name_ = mc.figpath + name

N = 20

if mc.anim_exist(name_):
    for i_N in xrange(N):
        im_ = mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0.), seed=seed+i_N)
        if i_N == 0:
            phase = 0.5 + 0. * im_[0, 0, :]#mc.N_X/2, mc.N_Y/2, :]
        #im += im_ - im_[mc.N_X/2, mc.N_Y/2, :] + phase
        im += im_ - im_[0, 0, :] + phase

    if play:
        mc.play(mc.rectif(im))
    else:
        mc.anim_save(mc.rectif(im), name_)
    #    mplayer figures/concentric.mpg -fs -loop 0
