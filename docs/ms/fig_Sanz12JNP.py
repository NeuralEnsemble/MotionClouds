#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script generates all figures and supplemental movies related to the publication:

@article{Sanz12,
        Author = {Sanz Leon, Paula and Vanzetta, Ivo and Masson, Guillaume S. and Perrinet, Laurent U.},
        Title = {Motion Clouds: Model-based stimulus synthesis of natural-like random textures for the study of motion perception},
        Year = {2012},
        Journal = {Journal of Neurophysiology},
        Url ={https://laurentperrinet.github.io/publication/sanz-12},
}



"""

import os
import numpy as np
import sys
sys.path.append('..')
import MotionClouds as mc

if not(os.path.isfile('figure1.pdf')):
    """
    Demonstration of going from a natural image to a random cloud by shuffling the phase in FFT space

    Figure 1: \caption{(\textit{A}) Top: a natural movie with the main motion component consisting of a
    horizontal, rightward full-field translation. Such a movie would be produced by an eye movement with
    constant mean velocity to the left (negative $x$ direction), plus some residual, centered jitter noise
    in the motion-compensated natural scene. We represent the movie as a cube, whose $(x,y,t=0)$ face
    corresponds to the first frame, the $(x,y=0,t)$ face shows the rightward translation motion as diagonal
    stripes. As a result of the horizontal motion direction, the $(x=54,y,t)$ face is a reflected image of
    the $(x,y,t=0)$ face, contracted or dilated depending on the amplitude of motion. The bottom panel shows
    the corresponding Fourier energy spectrum, as well as its projections onto three orthogonal planes. For
    any given point in frequency space, the energy value with respect to the maximum is coded by 6 discrete
    color iso-surfaces (i.e.: 90\%, 75\%, 50\%, 25\%, 11\% and 6\% of peak. The amplitude of the Fourier energy
    spectrum has been normalized to 1 in all panels and the same conventions used here apply to all following
    figures. (\textit{B}) to (\textit{C}): The image is progressively morphed (A through B to C) into a Random
    Phase Texture by perturbing independently the phase of each Fourier component.  (\textit{upper row}):
    Form is gradually lost in this process, whereas (\textit{lower row}): most motion energy information
    is preserved, as it is concentrated around the same speed plane in all three cases (the spectral
    envelopes are nearly identical).}

    """
    name = os.path.join(mc.figpath, 'morphing')
    vext = '.mp4'

    def randomize_phase(image, B_angle=0., vonmises=False, seed=None):
        Fz = np.fft.fftn(image)
        if B_angle > 0.:
            np.random.seed(seed=seed)
            if vonmises:
                Fz *= np.exp(1j * np.random.vonmises(mu=0. , kappa=1./B_angle, size=(N_X, N_Y, N_frame))) # mu = np.pi/2.
            else:
                Fz *= np.exp(1j * B_angle * np.random.randn(N_X, N_Y, N_frame))

        z = np.fft.ifftn(Fz).real
        return z

    def FTfilter(image, FTfilter):
        from scipy.fftpack import fftn, fftshift, ifftn, ifftshift
        from scipy import real
        FTimage = fftshift(fftn(image)) * FTfilter
        return real(ifftn(ifftshift(FTimage)))
    # pre-processing parameters
    white_f_0 = .5
    white_alpha = 1.4
    white_N = 0.01
    def whitening_filt(size, temporal=True, f_0=white_f_0, alpha=white_alpha, N=white_N):
        """
        Returns the whitening filter.

        Uses the low_pass filter used by (Olshausen, 98) where
        f_0 = 200 / 512

        parameters from Atick (p.240)
        f_0 = 22 c/deg in primates: the full image is approx 45 deg
        alpha makes the aspect change (1=diamond on the vert and hor, 2 = anisotropic)

        """
        fx, fy, ft = np.mgrid[-1:1:1j*size[0], -1:1:1j*size[1], -1:1:1j*size[2]]
        if temporal:
            rho = np.sqrt(fx**2+ fy**2 + ft**2)
        else:
            rho = np.sqrt(fx**2+ fy**2)
        low_pass = np.exp(-(rho/f_0)**alpha)
        K = (N**2 + rho**2)**.5 * low_pass
        return  K

    def whitening(image):
        """
        Returns the whitened sequence
        """
        K = whitening_filt(size=image.shape)
        white = FTfilter(image, K)
        # normalizing energy
        #    white /= white.max()# std() # np.sqrt(sum(I**2))
        return white

    def translate(image, vec):
        """
        Translate image by vec (in pixels)

        """
        u, v = vec

        # first translate by the integer value
        image = np.roll(np.roll(image, np.int(u), axis=0), np.int(v), axis=1)
        u -= np.int(u)
        v -= np.int(v)

        # sub-pixel translation
        from scipy import mgrid
        f_x, f_y = mgrid[-1:1:1j*image.shape[0], -1:1:1j*image.shape[1]]
        trans = np.exp(-1j*np.pi*(u*f_x + v*f_y))
        return FTfilter(image, trans)


    def translation(image, X_0=0., Y_0=0., V_X=.5, V_Y=0.0, V_noise=.0, width=2.):

        """

        >> pylab.imshow(concatenate((image[16,:,:],image[16,:,:]), axis = -1))

        """
        # translating the frame line_0
        movie = np.zeros(image.shape)
        V_X_, V_Y_ = V_X * (1+ np.random.randn()*V_noise), V_Y * (1+ np.random.randn()*V_noise)
        for i_frame, t in enumerate(np.linspace(0., 1., N_frame, endpoint=False)):
            V_X_, V_Y_ = V_X_ * (1+ np.random.randn()*V_noise), V_Y_ * (1+ np.random.randn()*V_noise)
            movie[:, :, i_frame] = translate((image[:, :, i_frame]), [(width/2 + X_0+t*V_X_*width + 1)*N_X/2., (width/2 + Y_0+t*V_Y_*width+1)*N_Y/2])
        return movie

    # I use a natural movie:
    N_frame, N_first = 32., 530
    #image = np.load('~/particles/movie/montypython.npy')[:, ::-1, N_first:(N_first+N_frame)]
    if False:#not os.path.exists('results/montypython.npy'):
        # Download the data
        import urllib
        print "Downloading data, Please Wait "
        opener = urllib.urlopen(
                'https://invibe.net/LaurentPerrinet/MotionClouds?action=AttachFile&do=get&target=montypython.npy')
        open('results/montypython.npy', 'wb').write(opener.read())

#     image = np.load('results/montypython.npy')[:, ::-1, N_first:(N_first+N_frame)]
    image = np.load('/Users/lolo/pool/science/MotionParticles/particles/movie/montypython.npy')[:, ::-1, N_first:(N_first+N_frame)]
    image -= image.mean()
    image /= np.abs(image).max()
    image += 1
    image /= 2.
    if not os.path.exists('results/montypython.npy.mp4'):
        mc.anim_save(image, 'results/montypython.npy', display=False, vext='.mp4')

    (N_X, N_Y, N_frame) = image.shape
    movie = translation(image)
    (N_X, N_Y, N_frame) = movie.shape
    movie = whitening(movie)

    print N_X, N_Y, N_frame
    fx, fy, ft = mc.get_grids(N_X, N_Y, N_frame)
    color = mc.envelope_color(fx, fy, ft) #
    z_noise = color*mc.envelope_speed(fx, fy, ft)
    movie_noise = mc.rectif(mc.random_cloud(z_noise))

    for B_angle in [1e-1, 1e0, 1e1]:
        name_ = name + '-B_angle-' + str(B_angle).replace('.', '_')
        movie_ = 1. * randomize_phase(movie, B_angle=B_angle) + .0 * movie_noise
        movie_ -= movie_.mean()
        movie_ /= np.abs(movie_).max()
        mc.cube(mc.rectif(movie_), name=name_ + '_cube')
        spectrum = np.absolute(np.fft.fftshift(np.fft.fftn(movie_)))# .real # **2 # + z_noise#
    #        img = img[:,::-1,:]
        spectrum /= spectrum.max()
        mc.visualize(spectrum, name=name_)

#print mc.N_X, mc.N_Y, mc.N_frame, mc.MAYAVI
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)

if not(os.path.isfile('figure2.pdf')):
    """
    Figure 2:  \caption{From an impulse to a Motion Cloud. (\textit{A}): The
    movie corresponding to a typical ``edge", i.e., a moving Gabor patch that
    corresponds to a localized grating. The Gabor patch being relatively small,
    for clarity, we zoomed 8 times into the non-zeros values of the image.
    (\textit{B}): By densely mixing multiple copies of the kernel shown in (A)
    at random positions, we obtain a Random Phase Texture (RPT), see
    Supplemental Movie 1.  (\textit{C}):  We show here the envelope of the
    Fourier transform of kernel $K$: inversely, $K$ is the impulse response in
    image space of the filter defined by this envelope. Due to the linearity of
    the Fourier transform, apart from a multiplicative constant that vanishes by
    normalizing the energy of the  RPT to $1$, the spectral envelope of the RPT
    in (B) is the same as the one of the kernel K shown in (A):
    $\mathcal{E}_{\bar{\beta}}=\mathcal{F}(K)$. Note that, the spectral
    energy envelope  of a ``classical" grating would result in a pair of
    Dirac delta functions centered on the peak of the patches in (C) (the
    orange ``hot-spots").  Motion Clouds are defined as the subset of such RPTs
    whos e main motion component is a full-field translations and thus
    characterized by spectral envelopes concentrated on a plane.}
    """
    name = 'grating'
    # making just an impulse
    name_ = os.path.join(mc.figpath, name + '-impulse')
    if mc.anim_exist(name_ + '_cube', vext=mc.ext):
        z = mc.envelope_gabor(fx, fy, ft)
        zoom = 8
        movie = mc.random_cloud(z, impulse=True)[:(mc.N_X/zoom), :(mc.N_Y/zoom), :(mc.N_frame/zoom)]
        movie = np.roll(movie, mc.N_frame/zoom, axis=2)
        mc.cube(mc.rectif(movie), name=name_ + '_cube', do_axis=False)#

if not(os.path.isfile('figure3.pdf')):
    """
    Figure 3:   \caption{Equivalent MC representations of some classical
    stimuli. (\textit{A}, \textit{top}): a narrow-orientation-bandwidth Motion
    Cloud produced only with vertically oriented kernels and a horizontal mean
    motion to the right (Supplemental Movie 1). (\textit{Bottom}): The spectral
    envelopes concentrated on a pair of patches centered on a constant speed
    surface. Note that this ``speed plane" is thin (as seen by the projection
    onto the ($f_x$,$f_t$) face), yet it has a finite thickness, resulting in
    small, local, jittering motion components. ({\textit{B}}) a Motion Cloud
    illustrating the aperture problem. (\textit{Top}): The stimulus, having
    oblique preferred orientation ($\theta=\frac{\pi}{4}$ and narrow bandwidth
    $B_{\theta}=\pi/36$) is moving horizontally and rightwards. However, the
    perceived speed direction in such a case is biased towards the oblique
    downwards, i.e., orthogonal to the orientation, consistently with the fact
    that the best speed plane is ambiguous to detect (Supplemental Movie 2).
    (\textit{C}): a low-coherence random-dot kinematogram-like Motion Cloud: its
    orientation and speed bandwidths, $B_{\theta}$ and $B_{V}$ respectively, are
    large, yielding a low-coherence stimulus in which no edges can be identified
    (Supplemental Movie 3).}

    """
    # A
    # the "standard MC" :  dense mixing of rightward gabor-like kernales
    z = mc.envelope_gabor(fx, fy, ft)
    mc.figures(z, name=os.path.join(mc.figpath, name), do_movie=False)

    # for div in [2, 4]:
    #     name_ =os.path.join(mc.figpath, name + '-theta-pi-over-' + str(div).replace('.', '_'))
    #     mc.figures_MC(fx, fy, ft, name_, theta=np.pi/div, do_movie=False)
    #
    # for sigma_div in [.5, 8, 16]:
    #     name_ = os.path.join(mc.figpath, name + '-B_theta-pi-over-' + str(sigma_div).replace('.', '_'))
    #     mc.figures_MC(fx, fy, ft, name_, B_theta=np.pi/sigma_div, do_movie=False)
    #
    # B
    grating = mc.envelope_gabor(fx, fy, ft, theta=np.pi/4, B_theta=np.pi/64)
    mc.figures(grating , name=os.path.join(mc.figpath, 'aperture'), do_movie=False)
    # C
    mc.figures_MC(fx, fy, ft, name=os.path.join(mc.figpath, 'grating-rdk'), B_theta=10., B_V=.5, B_sf=0.01, do_movie=False)

    #
    # for sf_0 in [0.01, 0.4]:
    #     name_ = figpath + name + '-sf_0-' + str(sf_0).replace('.', '_')
    #     mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0, do_movie=False)
    #
    # for B_sf in [0.05, 0.15, 0.4]:
    #     name_ = figpath + name + '-B_sf-' + str(B_sf).replace('.', '_')
    #     mc.figures_MC(fx, fy, ft, name_, B_sf=B_sf, do_movie=False)
    #

if not(os.path.isfile('SupplementalMovie1.mp4')):

    # Supplemental Movie 1
    # Standard Motion Cloud
    # By densely mixing multiple copies of corresponding to a typical moving
    # edge (i.e., a moving Gabor patch that corresponds to a localized grating)
    # at random positions, we obtain a Random Phase Texture (see Figure 3-A).
    z = mc.envelope_gabor(fx, fy, ft)
    mc.figures(z, name=os.path.join(mc.figpath, 'SupplementalMovie1'), do_figs=False)

if not(os.path.isfile('SupplementalMovie2.mp4')):
    # Supplemental movie 2
    # Motion Cloud with aperture problem
    # A Motion Cloud illustrating the aperture problem. It is a narrow bandwidth
    # Motion Cloud with a central orientation = pi/4 with respect to the mean
    # direction. The stimulus, having oblique preferred orientation and narrow
    # bandwidth is moving horizontally and rightwards. However, the perceived
    # speed direction in such a case is biased towards the oblique downwards,
    # i.e., orthogonal to the orientation, consistently with the fact that the
    # best speed plane is ambiguous to detect (see Figure 3-B).
    grating = mc.envelope_gabor(fx, fy, ft, theta=np.pi/4, B_theta=np.pi/64)
    mc.figures(grating , name=os.path.join(mc.figpath, 'SupplementalMovie2'), do_figs=False)

if not(os.path.isfile('SupplementalMovie3.mp4')):
    # Supplemental movie 3
    # Low-coherence random-dot kinematogram-like Motion Cloud
    # The orientation and speed bandwidths are large, yielding a low-coherence stimulus in which no edges can be identified (see Figure 3-C).
    mc.figures_MC(fx, fy, ft, name=os.path.join(mc.figpath, 'SupplementalMovie3'), B_theta=10., B_V=.5, B_sf=0.01, do_figs=False)

if not(os.path.isfile('figure4.pdf')):
    """
     Figure 4: \caption{Broadband vs. narrowband stimuli. From (\textit{A})
     through (\textit{B}) to (\textit{C}) the frequency bandwidth $B_{f}$
     increases, while all other parameters (such as $f_{0}$) are kept constant.
     The Motion Cloud with the broadest bandwidth is thought to best represent
     natural stimuli, since, as those, it contains many frequency components.
     (\textit{A}) $B_{f}=0.05$ (Supplemental Movie 4), (\textit{B}) $B_{f}=0.15$
     (Supplemental Movie 5) and (\textit{C}) $B_{f}=0.4$ (Supplemental Movie
     6).}

    """
    # for sf_0 in [0.1, 0.2, 0.4]:
    #     name_ = figpath + name + '-sf_0-' + str(sf_0).replace('.', '_')
    #     mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0, B_theta=10., do_movie=False)
    #
    for B_sf, i_SM in zip([0.05, 0.15, 0.4], [4, 5, 6]):
        name = os.path.join(mc.figpath, 'grating-B_sf-' + str(B_sf).replace('.', '_'))
        mc.figures_MC(fx, fy, ft, name, B_sf=B_sf, B_theta=10., do_movie=False)

        name = os.path.join(mc.figpath, 'SupplementalMovie' + str(i_SM))
        mc.figures_MC(fx, fy, ft, name, B_sf=B_sf, B_theta=10., do_figs=False)


# Supplemental movie 4
# Broadband vs. narrowband MCs
# Motion Cloud with a narrow spatial frequency bandwidth ($B_{sf}=0.05$), see Figure 4-A.

# Supplemental movie 5
# Broadband vs. narrowband s MC
# Motion Cloud with a medium narrow spatial frequency bandwidth ($B_{sf}=0.15$), see Figure 4-B.

# Supplemental movie 6
# Broadband vs. narrowband s MC
# Motion Cloud with a large spatial frequency bandwidth ($B_{sf}=0.40$), see Figure 4-.



# name = 'speed'
# for V_X in [0.5, 1.0]:
#     name_ = figpath + name + '-V_X-' + str(V_X).replace('.', '_')
#     mc.figures_MC(fx, fy, ft, name_, V_X=V_X, do_movie=False, B_theta=10., sf_0=.3, B_sf=.4)
#
#for V_Y in [1.0]:
#    name_ = figpath + name + '-V_Y-' + str(V_Y).replace('.', '_')
#    mc.figures_MC(fx, fy, ft, name_, V_X=0.0, V_Y=V_Y, do_movie=False, *options)
#
# for B_V in [0.1, 0.5]:#, 1.0]:
#     name_ = figpath + name + '-B_V-' + str(B_V).replace('.', '_')
#     mc.figures_MC(fx, fy, ft, name_, B_V=B_V, do_movie=False, B_theta=10., sf_0=.3, B_sf=.4)#, *options)
#
#
# name = 'color'
# for alpha in [0.0, 1.0, 2.0]:
#     # resp. white(0), pink(1), red(2) or brownian noise (see http://en.wikipedia.org/wiki/1/f_noise
#     name_ = figpath + name + '-alpha-' + str(alpha).replace('.', '_')
#     z = mc.envelope_color(fx, fy, ft, alpha)
#     mc.figures(z, name_, do_movie=False)
#
#
if not(os.path.isfile('figure5.pdf')):
    """
    Figure 5 : \caption{Competing Motion Clouds. (\textit{A}): A
    narrow-orientation-bandwidth Motion Cloud with explicit noise. A red noise
    envelope was added to the global envelop of a Motion Cloud with a bandwidth
    in the orientation domain (Supplemental Movie  7). (\textit{B}): Two Motion
    Clouds with same motion but different preferred orientation were added
    together, yielding a plaid-like Motion Cloud texture (Supplemental Movie 8).
    (\textit{C}): Two Motion Clouds with opposite velocity directions were
    added, yielding a texture similar to a ``counter-phase'' grating
    (Supplemental Movie 9). Note that the crossed shape in the $f_x-f_t$ plane
    is a signature of the opposite velocity directions, while two gratings with
    the same spatial frequency and in opposite directions would generate a
    flickering stimulus with energy concentrated on the $f_t$ plane.}

    """
    name = 'grating-noise'
    noise = mc.envelope_color(fx, fy, ft, alpha=1.)
    grating = mc.envelope_gabor(fx, fy, ft)
    mc.figures(1.e4*noise + grating, os.path.join(mc.figpath, name), do_movie=True)

    # Supplemental movie 7
    # A narrow-orientation-bandwidth Motion Cloud with explicit noise.
    # A red noise envelope was added to the global envelop of a Motion Cloud with a bandwidth in the orientation domain (see Figure 5-A).
    mc.figures(1.e4*noise + grating, name=os.path.join(mc.figpath, 'SupplementalMovie7'), do_figs=False)

    # B
    diag1 = mc.envelope_gabor(fx, fy, ft, theta=np.pi/4.)
    diag2 = mc.envelope_gabor(fx, fy, ft, theta=-np.pi/4.)
    mc.figures(diag1 + diag2, name=os.path.join(mc.figpath, 'plaid'), do_movie=True)

    # Supplemental movie 8
    # Plaid-like Motion Cloud.
    # Two Motion Clouds with same motion but different preferred orientation
    # were added together, yielding a plaid-like Motion Cloud texture (see
    # Figure 5-B).
    mc.figures(diag1 + diag2, name=os.path.join(mc.figpath, 'SupplementalMovie8'), do_figs=False)

    # C
    name = 'counterphase_grating'
    right = mc.envelope_gabor(fx, fy, ft, V_X=1 , B_theta=10.)
    left = mc.envelope_gabor(fx, fy, ft, V_X=-1. , B_theta=10.)
    # thanks to the addititivity of MCs
    mc.figures(left + right, name=os.path.join(mc.figpath, name), do_movie=True)

    #name = 'counterphase_grating2'
    #right = mc.envelope_gabor(fx, fy, ft, V_X=1 )
    #left = mc.envelope_gabor(fx, fy, ft, V_X=-1.)
    #mc.figures(left + right, name=os.path.join(mc.figpath, name), do_movie=False)
    #
    #name = 'counterphase_grating3'
    #grating = mc.envelope_gabor(fx, fy, ft, V_X=0, B_V=1.5, B_theta=np.pi/4, B_sf=0.01)
    #mc.figures(grating , name=os.path.join(mc.figpath, name), do_movie=False)

    # Supplemental movie 9
    # Counterphase-like MC
    # Two Motion Clouds with opposite velocity directions were added, yielding a
    # texture similar to a "counter-phase" grating. Note that the crossed shape
    # in the $f_x-f_t$ plane is a signature of the opposite velocity directions
    # (see Figure 5-C).
    mc.figures(left + right, name=os.path.join(mc.figpath, 'SupplementalMovie9'), do_figs=False)


# Stitching sub-figures together
# you have to install PIL: sudo pip install PIL
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import glob, os


def add_string_label(infile, outfile, label):
    # Define font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf", 48)#for Linux/Debian
    except:
        font = ImageFont.truetype("/usr/local/texlive/2013/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans.ttf", 48)#for MacOsX with MacTexLive
    for infilename, outfilename in zip(infile, outfile):
        img = Image.open(os.path.join(mc.figpath, infilename))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), label, (0, 0, 0), font=font)
        draw = ImageDraw.Draw(img)
        img.save(os.path.join(mc.figpath, outfilename))

if True:

    infile = ['morphing-B_angle-0_1_cube.png', 'grating-impulse_cube.png', 'grating_cube.png', 'grating-B_sf-0_05_cube.png', 'grating-noise_cube.png']
    outfile = ['figure1A.png', 'figure2A.png', 'figure3A.png', 'figure4A.png', 'figure5A.png']
    label = 'A'
    add_string_label(infile, outfile, label=label)

    infile = ['morphing-B_angle-1_0_cube.png', 'grating_cube.png', 'aperture_cube.png', 'grating-B_sf-0_15_cube.png', 'plaid_cube.png']
    label = 'B'
    outfile = ['figure1B.png', 'figure2B.png', 'figure3B.png', 'figure4B.png', 'figure5B.png']
    add_string_label(infile, outfile, label=label)

    infile = ['morphing-B_angle-10_0_cube.png', 'grating.png', 'grating-rdk_cube.png', 'grating-B_sf-0_4_cube.png', 'counterphase_grating_cube.png']
    label = 'C'
    outfile = ['figure1C.png', 'figure2C.png', 'figure3C.png', 'figure4C.png', 'figure5C.png']
    add_string_label(infile, outfile, label=label)



# ultimately, we use imagemagick montage commands to stitch figures together, see the Makefile



