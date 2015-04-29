#!/usr/bin/env python
"""
A smooth transition while changing parameters

(c) Laurent Perrinet - INT/CNRS

"""

import MotionClouds as mc
import numpy as np

name = 'smooth'
play = False
play = True

#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft)

name_ = mc.figpath + name

seed = 123456
B_sf_ = [0.025, 0.05, 0.1, 0.2, 0.4, 0.2, 0.1, 0.05]
im = np.empty(shape=(mc.N_X, mc.N_Y, 0))
name_ = mc.figpath + name + '-B_sf'
if mc.anim_exist(name_) or play:
    for i_sf, B_sf in enumerate(B_sf_):
        im_new = mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf), seed=seed)
        im = np.concatenate((im, im_new), axis=-1)

    if not(play): mc.anim_save(mc.rectif(im), name_)
    else: mc.play(im)
    # mplayer figures/smooth-B_sf.mpg -fs -loop 0

name_ += '_smooth'
if mc.anim_exist(name_):
    smooth = (ft - ft.min())/(ft.max() - ft.min()) # smoothly progress from 0. to 1.
    N = len(B_sf_)
    im =  np.empty(shape=(mc.N_X, mc.N_Y, 0))
    for i_sf, B_sf in enumerate(B_sf_):
        im_old = mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf), seed=seed)
        im_new = mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf_[(i_sf+1) % N]), seed=seed)
        im = np.concatenate((im, (1.-smooth)*im_old+smooth*im_new), axis=-1)

    if play:
        mc.play(mc.rectif(im))
    else:
        mc.anim_save(mc.rectif(im), name_)
    #    mplayer figures/smooth-B_sf_smooth.mpg -fs -loop 0
