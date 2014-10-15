#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of exploring the aperture problem using MotionClouds. Used to generate page:

http://invibe.net/LaurentPerrinet/SciBlog/2011-07-18

(c) Laurent Perrinet - INT/CNRS

"""

import numpy as np
import MotionClouds as mc

fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
name = 'SlippingClouds'
vext = '.gif'

B_sf = .3
B_theta = np.pi/32

# show example
table = """
#acl LaurentPerrinet,LaurentPerrinetGroup:read,write,delete,revert All:read
#format wiki
-----
= SlippingClouds: MotionClouds for exploring the aperture problem =

"""

theta1, theta2 = 0., np.pi/2.
diag = mc.envelope_gabor(fx, fy, ft, theta=theta2, V_X=np.cos(theta1), V_Y=np.sin(theta1), B_sf=B_sf, B_theta=B_theta)
name_ = name + '_iso'
mc.figures(diag, name_, vext=vext, seed=12234565)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'

theta1, theta2 = 0., np.pi/4.
diag = mc.envelope_gabor(fx, fy, ft, theta=theta2, V_X=np.cos(theta1), V_Y=np.sin(theta1), B_sf=B_sf, B_theta=B_theta)
name_ = name + '_diag'
mc.figures(diag, name_, vext=vext, seed=12234565)
table += '||{{attachment:' + name_ + '.png||width=100%}}||{{attachment:' + name_ + '_cube.png||width=100%}}||{{attachment:' + name_ + '.gif||width=100%}}||\n'

theta1, theta2 = 0., np.pi/2.
diag = mc.envelope_gabor(fx, fy, ft, theta=theta2, V_X=np.cos(theta1), V_Y=np.sin(theta1), B_sf=B_sf, B_theta=B_theta)
name_ = name + '_contra'
mc.figures(diag, name_, vext=vext, seed=12234565)
table += '||<width="33%">{{attachment:' + name_ + '.png||width=100%}}||<width="33%">{{attachment:' + name_ + '_cube.png||width=100%}}||<width="33%">{{attachment:' + name_ + '.gif||width=100%}}||\n'

table += '|||||| This figure shows how one can create !MotionCloud stimuli that have specific direction and orientation which are not necessarily perpendicular, components "slipping" relative to the motion. We generate !MotionClouds components with a strong selectivity toward one orientation. We show in different lines of this table respectively: Top) orientation perpendicular to orientation as in the standard case with a line, Middle) a 45° difference between both,  Bottom) orientation and direction are parallel. We created the aperture problem without any aperture! This can be shown by looking at envelopes in the Fourier space. To best infer motion, you would look fro the best speed plane that would go through each envelope. If these envelopes are rather tight, there are many different speed planes that can go through this envelope: motion is ambiguous. <<BR>> Columns represent isometric projections of a cube. The left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. The middle column shows an isometric view of the  faces of the movie cube. The first frame of the movie lies on the x-y plane, the x-t plane lies on the top face and motion direction is seen as diagonal lines on this face (vertical motion is similarly see in the y-t face). The third column displays the actual movie as an animation. ||\n'

table += '\n\n'

table += '== exploring different slipping angles ==\n'

# make just a line
N_orient = 8
#downscale= 2
#fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
theta = 0
for dtheta in np.linspace(0, np.pi/2, N_orient):
    name_ = name + '_dtheta_' + str(dtheta).replace('.', '_')
    theta1, theta2, B_theta = 0., np.pi/2., np.pi/32
    diag = mc.envelope_gabor(fx, fy, ft, theta=theta+dtheta, V_X=np.cos(theta), V_Y=np.sin(theta), B_sf=B_sf, B_theta=B_theta)
    mc.figures(diag, name_, vext=vext, seed=12234565)
    table += '||<width="50%">{{attachment:' + name_ + '.png||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||\n'

table += '||||  We display !SlippingClouds with different angle between direction and orientation. <<BR>> Left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. Right column of the table displays the actual movie as an animation.||\n'


table += '\n\n'

table += '== manipulating different ambiguities ==\n'

# make just a line
N_test = 8
#downscale= 2
#fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, mc.N_frame)
theta = 0
dtheta = np.pi/4
for B_theta_ in np.pi/8/np.arange(1, N_test+1):#np.linspace(0, np.pi/8, N_test)[1:]:
    name_ = name + '_B_theta_' + str(B_theta_).replace('.', '_')
    theta1, theta2, B_theta = 0., np.pi/2., np.pi/32
    diag = mc.envelope_gabor(fx, fy, ft, theta=theta+dtheta, V_X=np.cos(theta), V_Y=np.sin(theta), B_sf=B_sf, B_theta=B_theta_)
    mc.figures(diag, name_, vext=vext, seed=12234565)
    table += '||<width="50%">{{attachment:' + name_ + '.png||width=100%}}'
    table += '||<width="50%">{{attachment:' + name_ + '.gif||width=100%}}'
    table += '||\n'

table += '||||  The ambiguity in !SlippingClouds can be manipualted by changing B_theta for a given B_V. <<BR>> Left column displays iso-surfaces of the spectral envelope by displaying enclosing volumes at 5 different energy values with respect to the peak amplitude of the Fourier spectrum. Right column of the table displays the actual movie as an animation.||\n'

table += """
----
TagSciBlog TagMotion TagMotionClouds
"""


# TODO: automatic zip and uploading 
import os
os.system('zip zipped' + name + '.zip ' + name + '*')

print table

#perrinet@ghostrider:~$ rm /var/www/moin/perrinet/data/pages/SciBlog\(2f\)2011\(2d\)07\(2d\)18/attachments/SlippingClouds*
