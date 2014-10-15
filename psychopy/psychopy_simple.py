#!/usr/bin/env python
"""

A basic presentation in psychopy

(c) Laurent Perrinet - INT/CNRS


"""
# width and height of your screen
w, h = 1920, 1200
w, h = 2560, 1440

# width and height of the stimulus
w_stim, h_stim = 1024, 1024

loops = 4

import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft)
env = color *(mc.envelope_gabor(fx, fy, ft, V_X=1.) + mc.envelope_gabor(fx, fy, ft, V_X=-1.))
z = 2*mc.rectif(mc.random_cloud(env), contrast=.5) -1.

from psychopy import visual, core, event, logging
logging.console.setLevel(logging.DEBUG)

win = visual.Window([w, h], fullscr=True)
stim = visual.GratingStim(win, 
        size=(w_stim, h_stim), units='pix',
        interpolate=True,
        mask='gauss',
        autoLog=False)#this stim changes too much for autologging to be useful

for i_frame in range(mc.N_frame * loops):
    #creating a new stimulus every time
    stim.setTex(z[:, :, i_frame % mc.N_frame])
    stim.draw()
    win.flip()
