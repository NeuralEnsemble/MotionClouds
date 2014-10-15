#!/usr/bin/env python
"""

Superposition of MotionClouds to generate competing motions.

(c) Laurent Perrinet - INT/CNRS

"""
import numpy
import MotionClouds as mc
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

name = 'competing'
name_ = mc.figpath + name
if mc.anim_exist(name_):
    z = (.5*mc.envelope_gabor(fx, fy, ft, sf_0=0.2, V_X=-1.5)
         + .1*mc.envelope_gabor(fx, fy, ft, sf_0=0.4, V_X=.5)#, theta=numpy.pi/2.)
        )
    mc.figures(z, name_)

name = 'two_bands'
name_ = mc.figpath + name
if mc.anim_exist(name_):
    # and now selecting blobs:
    # one band
    one = mc.envelope_gabor(fx, fy, ft, B_theta=10.)
    # a second band
    two = mc.envelope_gabor(fx, fy, ft, sf_0=.9, B_theta=10.)

    mc.figures(one + two, name_)

# explore parameters
for sf_0 in [0.0, 0.1 , 0.2, 0.3, 0.8, 0.9]:
    name_ = mc.figpath + name + '-sf_0' + str(sf_0).replace('.', '_')
    if mc.anim_exist(name_):
        one = mc.envelope_gabor(fx, fy, ft, B_theta=10.)
        two = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_theta=10.)
        mc.figures(one + two, name_)


name = 'counterphase_grating'
name_ = mc.figpath + name
if mc.anim_exist(name_):
    right = mc.envelope_speed(fx, fy, ft, V_X=.5 )
    left = mc.envelope_speed(fx, fy, ft, V_X=-.5 )
    grating = mc.envelope_gabor(fx, fy, ft)
    z = grating * (left + right ) # thanks to the addititivity of MCs
    mc.figures(z, name_)

name = 'plaid'
name_ = mc.figpath + name
if mc.anim_exist(name):
    color = mc.envelope_color(fx, fy, ft)
    diag1 = mc.envelope_gabor(fx, fy, ft, theta=numpy.pi/4.)
    diag2 = mc.envelope_gabor(fx, fy, ft, theta=-numpy.pi/4.)
    z = color *(diag1 + diag2)
    mc.figures(z, name_)

# explore parameters
for V_X in [0., 0.5, 1.]:
    name_ = mc.figpath + name + '-V_X' + str(V_X).replace('.', '_')
    if mc.anim_exist(name_):
        diag = mc.envelope_gabor(fx, fy, ft, V_X=V_X)
        z = color *(diag + diag2)
        mc.figures(z, name_)

for V_Y in [0., 0.5, 1.]:
    name_ = mc.figpath + name + '-V_Y' + str(V_Y).replace('.', '_')
    if mc.anim_exist(name_):
        diag = mc.envelope_gabor(fx, fy, ft, V_Y=V_Y)
        z = color *(diag + diag2)
        mc.figures(z, name_)

for div in [1, 2, 3, 5, 8, 13 ]:
    name_ = mc.figpath + name + '-theta=pi-over-' + str(div).replace('.', '_')
    if mc.anim_exist(name_):
        z = color *(mc.envelope_gabor(fx, fy, ft, theta=numpy.pi/div) + mc.envelope_gabor(fx, fy, ft, theta=-numpy.pi/div))
        mc.figures(z, name_)
