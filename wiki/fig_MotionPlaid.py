#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of doing plaids using MotionClouds. 

Used to generate page:

http://invibe.net/LaurentPerrinet/SciBlog/2011-07-12

(c) Laurent Perrinet - INT/CNRS

"""

import numpy as np
import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

name = 'MotionPlaid'
vext = '.gif'

# show example
table = """
#acl LaurentPerrinet,LaurentPerrinetGroup:read,write,delete,revert All:
#format wiki
-----
= MotionPaids : from MotionClouds components to a plaid-like stimulus =

"""

theta1, theta2, B_theta = np.pi/4., -np.pi/4., np.pi/32
diag1 = mc.envelope_gabor(fx, fy, ft, theta=theta1, V_X=np.cos(theta1), V_Y=np.sin(theta1), B_theta=B_theta)
diag2 = mc.envelope_gabor(fx, fy, ft, theta=theta2, V_X=np.cos(theta2), V_Y=np.sin(theta2), B_theta=B_theta)
name_ = name + '_comp1'
mc.figures(diag1, name_, vext=vext, seed=12234565)
table += '||<width="33%">{{attachment:' + name_ + '.png||width=100%}}||<width="33%">{{attachment:' + name_ + '_cube.png||width=100%}}||<width="33%">{{attachment:' + name_ + '.gif||width=100%}}||\n'
name_ = name + '_comp2'
mc.figures(diag2, name_, vext=vext, seed=12234565)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'
name_ = name
mc.figures(diag1 + diag2, name, vext=vext, seed=12234565)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'
table += '||||||<align="justify">  This figure shows how one can create !MotionCloud stimuli that specifically target component and pattern cell. We show in the different lines of this table respectively: Top) one motion cloud component (with a strong selectivity toward the orientation perpendicular to direction) heading in the upper diagonal  Middle) a similar motion cloud component following the lower diagonal Bottom) the addition of both components: perceptually, the horizontal direction is predominant. <<BR>> Columns represent isometric projections of a cube. The left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. The middle column shows an isometric view of the  faces of the movie cube. The first frame of the movie lies on the x-y plane, the x-t plane lies on the top face and motion direction is seen as diagonal lines on this face (vertical motion is similarly see in the y-t face). The third column displays the actual movie as an animation. ||\n'

table += '\n\n'

table += '== exploring different component angles ==\n'


# make a grid
N_orient = 9
downscale = 4
fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
table += '|| '
for theta1 in np.linspace(0, 2*np.pi, N_orient)[::-1]:
    for theta2 in np.linspace(0, 2*np.pi, N_orient):
        name_ = name + 'theta1_' + str(theta1).replace('.', '_')
        name_ += '_theta2_' + str(theta2).replace('.', '_')
        diag1 = mc.envelope_gabor(fx, fy, ft, theta=theta1, V_X=np.cos(theta1), V_Y=np.sin(theta1), B_theta=B_theta)
        diag2 = mc.envelope_gabor(fx, fy, ft, theta=theta2, V_X=np.cos(theta2), V_Y=np.sin(theta2), B_theta=B_theta)
        mc.figures(diag1 + diag2, name_, vext=vext, seed=12234565)
        table += '{{attachment:' + name_ + '.gif||width=' + str(100/N_orient) +'%}}'
    table += '<<BR>>'
#    table += '||\n'
table += '||\n'

table += '||<align="justify">  As in (Rust, 06) we show in this table the concatenation of a table of ' + str(N_orient) + 'x' + str(N_orient) + ' !MotionPlaids where the angle of the components vary on respectively the horizontal and vertical axes. The diagonal from the bottom left to the top right corners show the addition of two component !MotionClouds of similar direction: They are therefore also intance of the same Motion Clouds and thus consist in a single component. As one gets further away from this diagonal, the angle between both component increases, as can be seen in the figure below. Note that first and last column are different instance of similar MotionClouds, just as first and last lines in in the table.||\n'

table += '\n\n'

# make just a line
N_orient = 8
downscale = 2
fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
theta = 0
#line1, line2 = '', ''
for dtheta in np.linspace(0, np.pi/2, N_orient):#, endpoint=False):
    name_ = name + 'dtheta_' + str(dtheta).replace('.', '_')
    diag1 = mc.envelope_gabor(fx, fy, ft, theta=theta + dtheta, V_X=np.cos(theta + dtheta), V_Y=np.sin(theta + dtheta), B_theta=B_theta)
    diag2 = mc.envelope_gabor(fx, fy, ft, theta=theta - dtheta, V_X=np.cos(theta - dtheta), V_Y=np.sin(theta - dtheta), B_theta=B_theta)
    mc.figures(diag1 + diag2, name_, vext=vext, seed=12234565)
#    line1 += '||<width="' + str(100/N_orient) +'%">{{attachment:' + name_ + '.png||width=100%}}'
#    line2 += '||{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.png||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||\n'

#table += line1 + '||\n' + line2 + '||\n' <-' + str(N_orient) + '>
table += '||||<align="justify">   For clarity, we display !MotionPlaids as the angle between both component increases from 0 to pi/2. <<BR>> Left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. Right column of the table displays the actual movie as an animation.||\n'

table += """
----
TagSciBlog TagMotion TagMotionClouds
"""


# TODO: automatic zip and uploading 
import os
os.system('zip zipped' + name + '.zip ' + name + '*')

print table