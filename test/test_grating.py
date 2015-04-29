#!/usr/bin/env python
"""

Exploring the orientation component of the envelope around a grating.

"""
try:
    if mc.notebook: print('we are in the notebook')
except:
    import os
    import MotionClouds as mc
    import numpy as np

name = 'grating'

#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
color = mc.envelope_color(fx, fy, ft)

z = color * mc.envelope_gabor(fx, fy, ft)
mc.figures(z, name)

# explore parameters
for sigma_div in [1, 2, 3, 5, 8, 13 ]:
    name_ = name + '-largeband-B_theta-pi-over-' + str(sigma_div).replace('.', '_')
    z = color * mc.envelope_gabor(fx, fy, ft, B_theta=np.pi/sigma_div)
    mc.figures(z, name_)

for div in [1, 2, 4, 3, 5, 8, 13, 20, 30]:
    name_ = name + '-theta-pi-over-' + str(div).replace('.', '_')
    z = color * mc.envelope_gabor(fx, fy, ft, theta=np.pi/div)
    mc.figures(z, name_)

V_X = 1.0
for sigma_div in [1, 2, 3, 5, 8, 13 ]:
    name_ = name + '-B_theta-pi-over-' + str(sigma_div).replace('.', '_') + '-V_X-' + str(V_X).replace('.', '_')
    z = color * mc.envelope_gabor(fx, fy, ft, V_X=V_X, B_theta=np.pi/sigma_div)
    mc.figures(z, name_)

for B_sf in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.8]:
    name_ = name + '-B_sf-' + str(B_sf).replace('.', '_')
    z = color * mc.envelope_gabor(fx, fy, ft, B_sf=B_sf)
    mc.figures(z, name_)
