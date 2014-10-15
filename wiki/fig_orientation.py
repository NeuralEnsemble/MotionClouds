#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of studying the role of orientation bandwidth using MotionClouds. 

Used to generate page:

http://invibe.net/LaurentPerrinet/SciBlog/2012-10-02


(c) Laurent Perrinet - INT/CNRS

"""

import numpy as np
import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

name = 'diego'
vext = '.gif'

# show example
table = """
#acl LaurentPerrinet,LaurentPerrinetGroup:read,write,delete,revert All:read
#format wiki
-----
=  Recruiting different population ratios in V1 using MotionClouds' orientation components =

"""


theta, B_theta_low, B_theta_high = np.pi/4., np.pi/32, 2*np.pi
B_V = 10.
seed=12234565

mc1 = mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0., theta=theta, B_V=B_V, B_theta=B_theta_low)
mc2 = mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0., theta=theta, B_V=B_V, B_theta=B_theta_high)

name_ = name + '_comp1'
mc.figures(mc1, name_, vext=vext, seed=seed)
table += '||<width="33%">{{attachment:' + name_ + '.png||width=100%}}||<width="33%">{{attachment:' + name_ + '_cube.png||width=100%}}||<width="33%">{{attachment:' + name_ + '.gif||width=100%}}||\n'
name_ = name + '_comp2'
mc.figures(mc2, name_, vext=vext, seed=seed)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'
table += '||||||<align="justify">  This figure shows how one can create !MotionCloud stimuli that specifically target different population in V1. We show in the two lines of this table  motion cloud component with a (Top) narrow orientation bandwith  (Bottom) a wide bandwitdh: perceptually, there is no predominant position or speed, just different orientation contents. <<BR>> Columns represent isometric projections of a cube. The left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. The middle column shows an isometric view of the  faces of the movie cube. The first frame of the movie lies on the x-y plane, the x-t plane lies on the top face and motion direction is seen as diagonal lines on this face (vertical motion is similarly see in the y-t face). The third column displays the actual movie as an animation. ||\n'

table += '\n\n'

table += '== exploring different orientation bandwidths ==\n'

# make just a line
downscale = 2
fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
#line1, line2 = '', ''
for B_theta in B_theta_high*2.**-np.arange(7):# np.linspace(B_theta_low, B_theta_high, N_orient):#, endpoint=False):
    name_ = name + 'B_theta_' + str(B_theta).replace('.', '_')
    mc_i = mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0., theta=theta, B_V=B_V, B_theta=B_theta)
    mc.figures(mc_i, name_, vext=vext, seed=seed)
    table += '||<width="50%">{{attachment:' + name_ + '.png||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||\n'

#table += line1 + '||\n' + line2 + '||\n' <-' + str(N_orient) + '>
table += '||||<align="justify">  Here, we display !MotionClouds as the orientation bandwidth increases from pi/32 to 2*pi. This should be equivalent in V1 to recruiting different ratios of neurons. <<BR>> Left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. Right column of the table displays the actual movie as an animation.||\n'

table += """
----
TagSciBlog TagMotion TagMotionClouds
"""


# TODO: automatic zip and uploading 
import os
os.system('zip zipped' + name + '.zip ' + name + '*')

print table