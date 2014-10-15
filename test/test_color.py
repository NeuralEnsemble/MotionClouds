#!/usr/bin/env python
"""

Testing differently colored noises.

"""

try:
    if mc.notebook: print('we are in the notebook')
except:
    import os
    import MotionClouds as mc
    import numpy as np

name = 'color'
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
z = mc.envelope_color(fx, fy, ft)
mc.figures(z, name)

# explore parameters
for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
    # resp. white(0), pink(1), red(2) or brownian noise (see http://en.wikipedia.org/wiki/1/f_noise
    name_ = name + '-alpha-' + str(alpha).replace('.', '_')
    z = mc.envelope_color(fx, fy, ft, alpha)
    mc.figures(z, name_)

for ft_0 in [0.125, 0.25, 0.5, 1., 2., 4., np.inf]:# time space scaling
    name_ = name + '-ft_0-' + str(ft_0).replace('.', '_')
    z = mc.envelope_color(fx, fy, ft, ft_0=ft_0)
    mc.figures(z, name_)

for contrast in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
    name_ = name + '-contrast-' + str(contrast).replace('.', '_')
    im = mc.rectif(mc.random_cloud(mc.envelope_color(fx, fy, ft)), contrast)
    mc.anim_save(im, os.path.join(mc.figpath, name_), display=False)

for contrast in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
    name_ = name + '-energy_contrast-' + str(contrast).replace('.', '_')
    im = mc.rectif(mc.random_cloud(mc.envelope_color(fx, fy, ft)), contrast, method='energy')
    mc.anim_save(im, os.path.join(mc.figpath, name_), display=False)

for seed in [123456 + step for step in range(7)]:
    name_ = name + '-seed-' + str(seed)
    mc.anim_save(mc.rectif(mc.random_cloud(mc.envelope_color(fx, fy, ft), seed=seed)), os.path.join(mc.figpath, name_), display=False)

for size in range(5, 7):
    N_X, N_Y, N_frame = 2**size, 2**size, 2**size
    fx, fy, ft = mc.get_grids(N_X, N_Y, N_frame)
    ft_0 = N_X/float(N_frame)
    name_ = name + '-size-' + str(size).replace('.', '_')
    z = mc.envelope_color(fx, fy, ft, ft_0=ft_0)
    mc.figures(z, name_)

for size in range(5, 7):
    N_frame = 2**size
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, N_frame)
    ft_0 = N_X/float(N_frame)
    name_ = name + '-size_T-' + str(size).replace('.', '_')
    z = mc.envelope_color(fx, fy, ft, ft_0=ft_0)
    mc.figures(z, name_, do_figs=False)
