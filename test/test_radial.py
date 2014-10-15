#!/usr/bin/env python
"""

We explore the radial envelope in MotionClouds.

"""
try:
    if mc.notebook: print('we are in the notebook')
except:
    import os
    import MotionClouds as mc
    import numpy as np

name = 'radial'
B_theta = np.pi/8.
#initialize
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

mc.figures_MC(fx, fy, ft, name, B_theta=B_theta)
verbose = False

# explore parameters
for B_sf in np.logspace(-2., 0.1, 5):#[0.0, 0.1, 0.2, 0.3, 0.8]:
    name_ = name + '-B_sf-' + str(B_sf).replace('.', '_')
    mc.figures_MC(fx, fy, ft, name_, B_sf=B_sf, B_theta=B_theta, verbose=verbose)

for B_V in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    name_ = name + '-B_V-' + str(B_V).replace('.', '_')
    mc.figures_MC(fx, fy, ft, name_, B_V=B_V, B_theta=B_theta, verbose=verbose)

for sf_0 in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]:
    name_ = name + '-sf_0-' + str(sf_0).replace('.', '_')
    mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0, B_theta=B_theta, verbose=verbose)

for sf_0 in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]:
    name_ = name + '-sf_0_nologgabor-' + str(sf_0).replace('.', '_')
    mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0, B_theta=B_theta, loggabor=False, verbose=verbose)

# for seed in [123456, 123457, None, None]:
#     name_ = name + '-seed-' + str(seed)
#     mc.figures_MC(fx, fy, ft, name_, seed=seed, B_theta=B_theta, verbose=verbose)
# 
for V_X in [0., 0.5, 1., -1.]:
    name_ = name + '-V_X-' + str(V_X).replace('.', '_')
    mc.figures_MC(fx, fy, ft, name_, V_X=V_X, B_theta=B_theta, verbose=verbose)
