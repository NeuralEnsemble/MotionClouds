#!/usr/bin/env python
"""
experiment_VSDI.py
Experiment designed for optical imaging showing a conversion of size from Fourier 
to screen coordinates and an export to a zipped file conaining BMP files (as the
video card has limited memory)

(c) Paula Sanz Leon - INT/CNRS


"""
import os
import scipy
import MotionClouds as mc
import numpy as np

# uncomment to preview movies
# vext, display = None, True

#------------------------- Zipped grating ------------------------------- #

name = 'zipped_grating'

#initialize - 
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
sf_0 = 0.2
B_sf = sf_0 / 4.
alpha = 1.0
V_X = 0.5
B_V = V_X / 10.

name_ = mc.figpath + name
for seed in range(424242, 424242+8):
    name_ = mc.figpath + name + '-seed-' + str(seed)
    mc.figures_MC(fx, fy, ft, name_, B_V=B_V, sf_0=sf_0, B_sf=B_sf, V_X=V_X, theta=np.pi/4., alpha=alpha, seed=seed, vext='.zip')

#-------------------- Narrowband vs Braodband experiment ---------------- #
vext = '.mpg'
#vext = '.mat'
#vext = '.zip'
#display = False

# Initialize frequency cube
N_X = 640.
N_Y = 480.
N_frame = 30. # a full period in time frames
fx, fy, ft = mc.get_grids(N_X, N_Y, N_frame)

# Experimental constants 
contrast = 0.5
seeds = 1
VA = 38.0958       # VSG constants for a viewing ditance 570 mm. 
framerate = 50.    # Refreshing rate in [Hz]
T = 0.6            # Stimulus duration [s] 
f_d = 0.5         # Desired spatial frequency [cpd]

# Clouds parameters      
B_V = 0.2     # BW temporal frequency (speed plane thickness)
B_sf = 0.15   # BW spatial frequency
theta = 0.0   # Central orientation
B_theta = np.pi/12 # BW orientation
verbose = False
alpha = 1.0

# Get normalised units
sf_0=0.1
V_X=0.5

def physicalUnits2discreteUnits(sf0_cpd, Bsf_cpd, v_dps, B_dps,pixelPitch,viewingDistance,frameRate):
    %   % laptop monitor
    %   pixelPitch      = 0.22/10; % in cm
    %   viewingDistance = 50;
    %   sf0_cpd         = 4 ;
    %   Bsf_cpd         = 1 ;
    %   v_dps           = 4 ;
    %   B_dps           = 1 ;
    %   frameRate       = 20 ;
    % convert to machine units
    cmPerDegree    = 2*viewingDistance*tand(1/2)
    pxPerDegree    = cmPerDegree/pixelPitch
    sf0            = sf0_cpd/pxPerDegree
    Bsf            = Bsf_cpd/pxPerDegree

    v              = v_dps/frameRate*pxPerDegree
    Bv             = B_dps/frameRate*pxPerDegree
    return sf0, Bsf, v, Bv


% take monitor parameters
% planar display
monitorRefresh = 60 ;
pixelPitch     = 0.2865/10 ;% pixelsize in cm

% experiments parameter
viewingDistance = 70 ;

frameRefresh   = monitorRefresh/3 ;
stimModRate    = 1 ;
numberOfReps   = 12 ;
framesPerMod   = frameRefresh/stimModRate/2 ; 


# Masks
# gaussian mask
sigma_mask_x = 0.15
sigma_mask_y = 0.2
x, y, t = mc.get_grids(N_X, N_Y, N_frame)
n_x, n_y = N_X, N_Y
gauss = np.exp(-(((x-172./n_x)**2/(2*sigma_mask_x**2)) + (((y-108./n_y)**2)/(2*sigma_mask_y**2))))


def tukey(n, r=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a 
    cosine lobe of width r * N / 2 that is convolved with a rectangle window of width 
    (1 - r / 2). At r = 1 it becomes rectangular, and at r = 0 it becomes a Hann window.
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    '''
    # Special cases
    if r <= 0:
        return np.ones(n.shape) #rectangular window
    elif r >= 1:
        return np.hanning(n.shape)

    # Normal case
    x = np.linspace(0, 1, n)
    w = np.ones(x.shape)

    # first condition 0 <= x < r/2
    first_condition = x<r/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/r * (x[first_condition] - r/2) ))

    # second condition already taken care of

    # third condition 1 - r / 2 <= x <= 1
    third_condition = x>=(1 - r/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/r * (x[third_condition] - 1 + r/2))) 

    return w		

# Tukey mask - fading effect
tw_x = tukey(n=n_x, r=0.15)
tw_y = tukey(n=n_y, r=0.15)
w = np.tile(((np.outer(tw_y,tw_x))), (N_frame,1,1))
tukey_mask = w.T


# Get Random Clouds
name_ = mc.figpath + name

for seed in [123456 + step for step in range(seeds)]:
	name__ = mc.figpath + name + '-seed-' + str(seed) + '-sf0-' + str(sf_0).replace('.', '_') + '-V_X-' + str(V_X).replace('.', '_')
        # broadband 
        z = mc.envelope_gabor(fx, fy, ft, name_, B_sf=Bsf, sf_0=sf_0, theta=theta, B_V=B_V, B_theta = B_theta, alpha=alpha)
        movie = mc.figures(z, name=None, vext=vext, seed=seed, masking=True)    
        for label, mask in zip(['_mask', '_tukey_mask'], [gauss, tukey_mask]):
            name_ = name__ + '-cloud-' + label
            if anim_exist(name_): 
                movie = mc.rectif(movie*mask)
                mc.anim_save(movie, name_, display=False, vext=vext)
        
       # narrowband 
        z = mc.envelope(fx, fy, ft, name_, B_sf=B_sf/10., sf_0=sf_0, theta=theta, B_V=B_V, B_theta=B_theta, alpha=alpha)
        movie = mc.figures(z, name=None, vext=vext, seed=seed, masking=True)
        for label, mask in zip(['_mask', 'tukey_mask'], [gauss, tukey_mask]):
            name_ = name__ + '-blob-' + label
            if anim_exist(name_):
                movie = mc.rectif(movie*mask)
                mc.anim_save(movie, name_, display=False, vext=vext)     			
       
