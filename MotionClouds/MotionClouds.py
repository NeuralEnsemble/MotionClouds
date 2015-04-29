# /usr/bin/env python
# -*- coding: utf8 -*-
"""

Main script for generating Motion Clouds

(c) Laurent Perrinet - INT/CNRS

Motion Clouds (keyword) parameters:
size    -- power of two to define the frame size (N_X, N_Y)
size_T  -- power of two to define the number of frames (N_frame)
N_X     -- frame size horizontal dimension [px]
N_Y     -- frame size vertical dimension [px]
N_frame -- number of frames [frames] (a full period in time frames)
alpha   -- exponent for the color envelope.
sf_0    -- mean spatial frequency relative to the sampling frequency.
ft_0    -- spatiotemporal scaling factor.
B_sf    -- spatial frequency bandwidth
V_X     -- horizontal speed component
V_Y     -- vertical speed component
B_V     -- speed bandwidth
theta   -- mean orientation of the Gabor kernel
B_theta -- orientation bandwidth
loggabor-- (boolean) if True it uses a log-Gabor kernel (instead of the traditional gabor)

Display parameters:

vext       -- movie format. Stimulus can be saved as a 3D (x-y-t) multimedia file: .mpg or .mp4 movie, .mat array, .zip folder with a frame sequence.
ext        -- frame image format.
T_movie    -- movie duration [s].
fps        -- frame per seconds

"""

import os
import numpy as np
from .param import *

def get_grids(N_X, N_Y, N_frame):
    """
        Use that function to define a reference outline for envelopes in Fourier space.
        In general, it is more efficient to define dimensions as powers of 2.
        output is always  of even size..

    """
    fx, fy, ft = np.mgrid[(-N_X//2):((N_X-1)//2 + 1), (-N_Y//2):((N_Y-1)//2 + 1), (-N_frame//2):((N_frame-1)//2 + 1)]
    fx, fy, ft = fx*1./N_X, fy*1./N_Y, ft*1./N_frame
    return fx, fy, ft

def frequency_radius(fx, fy, ft, ft_0=ft_0):
    """
     Returns the frequency radius. To see the effect of the scaling factor run
     'test_color.py'

    """
    N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
    if ft_0==np.inf:
        R2 = fx**2 + fy**2
        R2[N_X//2 , N_Y//2 , :] = np.inf
    else:
        R2 = fx**2 + fy**2 + (ft/ft_0)**2 # cf . Paul Schrater 00
        R2[N_X//2 , N_Y//2 , N_frame//2 ] = np.inf
    return np.sqrt(R2)

def envelope_color(fx, fy, ft, alpha=alpha, ft_0=ft_0):
    """
    Returns the color envelope.
    Run 'test_color.py' to see the effect of alpha
    alpha = 0 white
    alpha = 1 pink
    alpha = 2 red/brownian
    (see http://en.wikipedia.org/wiki/1/f_noise )
    """
    N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
    f_radius = frequency_radius(fx, fy, ft, ft_0=ft_0)**alpha
    if ft_0==np.inf:
        f_radius[N_X//2 , N_Y//2 , : ] = np.inf
    else:
        f_radius[N_X//2 , N_Y//2 , N_frame//2 ] = np.inf
    return 1. / f_radius

def envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, ft_0=ft_0, loggabor=loggabor):
    """
    Returns the radial frequency envelope:

    selects a sphere around a preferred frequency with a shell width B_sf.
    Run 'test_radial.py' to see the explore the effect of sf_0 and B_sf

    """
    if sf_0 == 0.: return 1.
    if loggabor:
        # see http://en.wikipedia.org/wiki/Log-normal_distribution
        fr = frequency_radius(fx, fy, ft, ft_0=ft_0)
        env = 1./fr*np.exp(-.5*(np.log(fr/sf_0)**2)/(np.log((sf_0+B_sf)/sf_0)**2))
        return env
    else:
        return np.exp(-.5*(frequency_radius(fx, fy, ft, ft_0=ft_0) - sf_0)**2/B_sf**2)

def envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V):
    """
    Returns the speed envelope:
    selects the plane corresponding to the speed (V_X, V_Y) with some thickness B_V

    * (V_X, V_Y) = (0,1) is downward and  (V_X, V_Y) = (1, 0) is rightward in the movie.
    * A speed of V_X=1 corresponds to an average displacement of 1/N_X per frame.
    To achieve one spatial period in one temporal period, you should scale by
    V_scale = N_X/float(N_frame)
    If N_X=N_Y=N_frame and V=1, then it is one spatial period in one temporal
    period. It can be seen along the diagonal in the fx-ft face of the MC cube.

    Run 'test_speed.py' to explore the speed parameters

    """
    env = np.exp(-.5*((ft+fx*V_X+fy*V_Y))**2/(B_V*frequency_radius(fx, fy, ft, ft_0=ft_0))**2)
    return env

def envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta):
    """
    Returns the orientation envelope:
    selects one central orientation theta, B_theta the spread
    We use a von-Mises distribution on the orientation.

    Run 'test_orientation.py' to see the effect of changing theta and B_theta.
    """
    if not(B_theta is np.inf):
        angle = np.arctan2(fy, fx)
        envelope_dir = np.exp(np.cos(2*(angle-theta))/B_theta)
        return envelope_dir
    else: # for large bandwidth, returns a strictly flat envelope
        return 1.

def envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha):
    """
    Returns the Motion Cloud kernel, that is the product of:
        * a speed envelope
        * an orientation envelope
        * an orientation envelope

    """
    envelope = envelope_color(fx, fy, ft, alpha=alpha)
    envelope *= envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta)
    envelope *= envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor)
    envelope *= envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V)
    return envelope

def random_cloud(envelope, seed=None, impulse=False, do_amp=False, threshold=1.e-3):
    """
    Returns a Motion Cloud movie as a 3D matrix from a given envelope.

    It first creates a random phase spectrum, multiplies with the envelope and
    then it computes the inverse FFT to obtain the spatiotemporal stimulus.

    Options are:
     * use a specific seed to specify the RNG's seed,
     * test the impulse response of the kernel by setting impulse to True
     * test the effect of randomizing amplitudes too by setting do_amp to True
shape

    # TODO : issue a warning if more than 10% of the energy of the envelope falls off the Fourier cube
    # TODO : use a safety sphere to ensure all orientations are evenly chosen

    """

    (N_X, N_Y, N_frame) = envelope.shape
    amps = 1.
    if impulse:
        fx, fy, ft = get_grids(N_X, N_Y, N_frame)
        phase = -2*np.pi*(N_X/2*fx + N_Y/2*fy + N_frame/2*ft)
    else:
        np.random.seed(seed=seed)
        phase = 2 * np.pi * np.random.rand(N_X, N_Y, N_frame)
        if do_amp:
            # see Galerne, B., Gousseau, Y. & Morel, J.-M. Random phase textures: Theory and synthesis. IEEE Transactions in Image Processing (2010). URL http://www.biomedsearch.com/nih/Random-Phase-Textures-Theory-Synthesis/20550995.html. (basically, they conclude "Even though the two processes ADSN and RPN have different Fourier modulus distributions (see Section 4), they produce visually similar results when applied to natural images as shown by Fig. 11.")
            amps = np.random.randn(N_X, N_Y, N_frame)

    Fz = amps * envelope * np.exp(1j * phase)

    # centering the spectrum
    Fz = np.fft.ifftshift(Fz)
    Fz[0, 0, 0] = 0.
    z = np.fft.ifftn((Fz)).real
    return z

