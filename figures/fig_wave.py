#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A bit of fun with gravitational waves

Used to generate page:

    http://invibe.net/LaurentPerrinet/SciBlog/2012-12-19

Source code on:
    
    https://raw.github.com/NeuralEnsemble/MotionClouds/master/fig_orientation.py

(c) Laurent Perrinet - INT/CNRS

"""

import numpy as np
import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

name = 'waves'
vext = '.gif'

# show example
table = """
#acl LaurentPerrinet,LaurentPerrinetGroup:read,write,delete,revert All:read
#format wiki
-----
= A bit of fun with gravitational waves =

"""

def envelope_gravitational(fx, fy, ft, B_v, g=.05):
    """
     Gravitational envelope:
     selects the plane corresponding to the speed (V_X, V_Y) with some thickness B_V

    """
#    env = np.exp(-.5*((ft**2/g+fx*V_X+fy*V_Y))**2/(B_V*mc.frequency_radius(fx, fy, ft, ft_0=1.))**2)
#    env += np.exp(-.5*((-ft**2/g+fx*V_X+fy*V_Y))**2/(B_V*mc.frequency_radius(fx, fy, ft, ft_0=1.))**2)
    env = np.exp(-.5*((ft**2+np.sqrt(g*(fx**2+fy**2)))**2/(B_v*mc.frequency_radius(fx, fy, ft, ft_0=1.))**2))
    env += np.exp(-.5*((-ft**2+np.sqrt(g*(fx**2+fy**2)))**2/(B_v*mc.frequency_radius(fx, fy, ft, ft_0=1.))**2))
    return env

def envelope_gabor_wave(fx, fy, ft, V_X=mc.V_X, V_Y=mc.V_Y,
                    B_V=mc.B_V, B_v=1., sf_0=mc.sf_0, B_sf=mc.B_sf, loggabor=mc.loggabor,
                    theta=mc.theta, B_theta=mc.B_theta, alpha=mc.alpha):
    """
    Returns the Motion Cloud kernel

    """
    envelope = mc.envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha)
    envelope *= envelope_gravitational(fx, fy, ft, B_v=B_v)
    return envelope

theta, B_theta, B_v_low, B_v_high = 0., np.pi/3., .025, .1
loggabor, alpha, sf_0, B_sf, B_V = True, 0., .07, .1, .6
seed = 12234565

name_ = name + '_low'
if mc.anim_exist(name_):
    mc1 = envelope_gabor_wave(fx, fy, ft, V_X=1., V_Y=0., B_v=B_v_low, B_V=B_V, theta=theta, B_theta=B_theta, sf_0=sf_0, B_sf=B_sf, alpha=alpha)
    mc.figures(mc1, name_, vext=vext, seed=seed)
table += '||<width="33%">{{attachment:' + name_ + '.png||width=100%}}||<width="33%">{{attachment:' + name_ + '_cube.png||width=100%}}||<width="33%">{{attachment:' + name_ + '.gif||width=100%}}||\n'
name_ = name + '_high'
if mc.anim_exist(name_):
    mc2 = envelope_gabor_wave(fx, fy, ft, V_X=1., V_Y=0., B_v=B_v_high, B_V=B_V, theta=theta, B_theta=B_theta, sf_0=sf_0, B_sf=B_sf, alpha=alpha)
    mc.figures(mc2, name_, vext=vext, seed=seed)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'
table += '||||||<align="justify">  This figure shows how one can create stimuli similar to !MotionCloud stimuli that have a dstribution according to the laws of gravitational waves (that is with temporal frequency proportional to the square root of spatial frequency for deep water conditions -- in the shallow water conditions, normal MCs are a good approximation). Note that phase speed (following a maximum) is twice faster than group speed (following a dot). The two lines correspond to different bandwitdhs for the spread around the physical law (width of the manifold). <<BR>> Columns represent isometric projections of a cube. The left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. The middle column shows an isometric view of the  faces of the movie cube. The first frame of the movie lies on the x-y plane, the x-t plane lies on the top face and motion direction is seen as diagonal lines on this face (vertical motion is similarly see in the y-t face). The third column displays the actual movie as an animation. ||\n'

table += '\n\n'

if True:#try:
    from enthought.mayavi import mlab
    downscale = 1
    x, y = np.mgrid[(-mc.N_X//2):((mc.N_X-1)//2 + 1), (-mc.N_Y//2):((mc.N_Y-1)//2 + 1)]
    fig = mlab.figure(1, size=(1600/downscale, 1200/downscale), fgcolor=(1, 1, 1), bgcolor=(0.5, 0.5, 0.5))
    z = mc.rectif(mc.random_cloud(mc1))
    for frame in range(mc.N_frame):
        mlab.clf()
        #s = mlab.SurfRegular(mc1[:, :, frame])
        #s = mlab.surf(mc1[:, :, frame])
        # http://docs.enthought.com/mayavi/mayavi/auto/example_wigner.html
        surf_ = mlab.surf(x, y, z[:, :, frame], colormap='bone', warp_scale=15, vmin=-1., vmax=1.)
        mlab.outline(surf_, color=(.7, .7, .7))
#fig.add(s)
        view = mlab.view(azimuth=290., elevation=60, distance='auto', focalpoint='auto')
        mlab.savefig('_MC_%03d.png' % frame, magnification=0.5)
    import os
    os.system('ffmpeg -v 0 -y -sameq  -loop_output 0 -i _MC_%03d.png  MC_wave.mpg')
    os.system('convert  _MC_%03d.png  MC_wave.gif')
#except:
#    print('TODO: visualise the free surface in 3D')

table += '== exploring different orientation bandwidths ==\n'

# make just a line
downscale = 2
fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
#line1, line2 = '', ''
for B_theta in (2*np.pi)*2.**-np.arange(7):
    name_ = name + '_B_theta_' + str(B_theta).replace('.', '_')
    if mc.anim_exist(name_):
        mc_i = envelope_gabor_wave(fx, fy, ft, V_X=1., V_Y=0., B_V=B_V, B_v=B_v_low, theta=theta, B_theta=B_theta, sf_0=sf_0, B_sf=B_sf, alpha=alpha)
        mc.figures(mc_i, name_, vext=vext, seed=seed)
    table += '||<width="50%">{{attachment:' + name_ + '.png||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||\n'

#table += line1 + '||\n' + line2 + '||\n' <-' + str(N_orient) + '>
table += '||||<align="justify">  Here, we display these modified !MotionClouds as the orientation bandwidth increases from pi/32 to 2*pi. This should be equivalent in the ocean to different dispersion of the wind that generates the waves. <<BR>> Left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. Right column of the table displays the actual movie as an animation.||\n'

table += """
----
TagSciBlog TagMotion TagMotionClouds
"""

import os
os.system('zip zipped' + name + '.zip ' + name + '*')

print table
