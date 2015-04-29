#!/usr/bin/env python
"""

Testing the role of different parameters in ther speed envelope.

"""

try:
    if mc.notebook: print('we are in the notebook')
except:
    import os
    import MotionClouds as mc
    import numpy as np


#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft) #

name = 'speed'
# now selects in preference the plane corresponding to the speed with some thickness
z = color*mc.envelope_speed(fx, fy, ft)
mc.figures(z, name)

# explore parameters
for V_X in [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 4.0]:
    name_ = name + '-V_X-' + str(V_X).replace('.', '_')
    z = color * mc.envelope_speed(fx, fy, ft, V_X=V_X)
    mc.figures(z, name_)

for V_Y in [-1.0, -0.5, 0.5, 1.0, 2.0]:
    name_ = name + '-V_Y-' + str(V_Y).replace('.', '_')
    z = color * mc.envelope_speed(fx, fy, ft, V_X=0.0, V_Y=V_Y)
    mc.figures(z, name_)

for B_V in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    name_ = name + '-B_V-' + str(B_V).replace('.', '_')
    z = color * mc.envelope_speed(fx, fy, ft, B_V=B_V)
    mc.figures(z, name_)
