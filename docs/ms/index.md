.. title: Motion Clouds
.. slug: index
.. date: 2014/01/15 11:20:10
.. tags: motion-clouds, python
.. link:
.. description: home page for Motion Clouds
.. type: text

**MotionClouds** are random dynamic stimuli optimized to study motion perception.

In particular, these stimuli can be made closer to naturalistic textures compared to usual stimuli such as gratings and random-dot kinetograms. These have controlled information content: We simplified the definition to parametrically define these "Motion Clouds" around the most prevalent feature axis (mean and bandwith): direction, scale (spatial frequency), orientation. These scripts implement a framework to generate these random texture movies.

<center><img src="https://neuralensemble.github.io/MotionClouds/files/grating.gif" width="50%"></center>

The description of this method was published in:

* Paula S. Leon, Ivo Vanzetta, Guillaume S. Masson, Laurent U. Perrinet. _Motion Clouds: Model-based stimulus synthesis of natural-like random textures for the study of motion perception._ [**Journal of Neurophysiology**](http://jn.physiology.org/content/early/2012/03/10/jn.00737.2011), 107(11):3217--3226, 2012  [URL](https://laurentperrinet.github.io/publication/sanz-12) - [preprint](https://neuralensemble.github.io/MotionClouds/ms/MotionClouds.pdf) - [Supplementary Information](https://neuralensemble.github.io/MotionClouds/ms/MotionClouds_Supplementary.pdf) - [Supplementary Videos](https://neuralensemble.github.io/MotionClouds/ms/MotionClouds_VideoFigures.pdf)

and recently in

* Jonathan Vacher, Andrew Isaac Meso, Laurent U Perrinet, Gabriel Peyré. _Biologically Inspired Dynamic Textures for Probing Motion Perception_. In [**Advances in Neural Information Processing Systems**](https://laurentperrinet.github.io/publication/vacher-15-nips) 28, pages 1909--1917. Curran Associates, Inc., 2015; [arXiv version with supplementary Information](http://arxiv.org/abs/1511.02705)

While this method was used in the following paper:

* Claudio Simoncini, Laurent U. Perrinet, Anna Montagnini, Pascal Mamassian, Guillaume S. Masson. _More is not always better: dissociation between perception and action explained by adaptive gain control._ [**Nature Neuroscience**](http://www.nature.com/neuro/journal/v15/n11/full/nn.3229.html), 2012 [URL](https://laurentperrinet.github.io/publication/simoncini-12)

This work was supported by ANR project ["ANR Speed"](https://laurentperrinet.github.io/project/anr-speed/) ANR-13-BSHS2-0006.

<img src="http://www.agence-nationale-recherche.fr/images/logos/ANR07-396.gif" width="30%">

This work was supported by the European Union project Number FP7-269921, ``BrainScaleS'' (Brain-inspired multiscale computation in neuromorphic hybrid systems), an EU FET-Proactive FP7 funded research project. The project started on 1 January 2011. It is a collaboration of 18 research groups from 10 European countries.

<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/e/e2/Public--BrainScalesLogo.svg/100px-Public--BrainScalesLogo.svg.png" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/8/88/Public--FET--FETTreeLogo.jpg/70px-Public--FET--FETTreeLogo.jpg" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/3/3b/Public--EU-FP7Logo.gif/90px-Public--EU-FP7Logo.gif" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/5/5b/Public--EU-Logo.gif/90px-Public--EU-Logo.gif" width="10%">

***
## Code example - [demo](https://neuralensemble.github.io/MotionClouds/posts/testing_components.html)

Motion Clouds are built using a collection of scripts that provides a simple way of generating complex stimuli suitable for neuroscience and psychophysics experiments. It is meant to be an open-source package that can be combined with other packages such as PsychoPy or NeuroTools.

All functions are implemented in one main script called `MotionClouds.py` that handles the Fourier cube, the envelope functions as well as the random phase generation and all Fourier related processing. Additionally, all the auxiliary visualization tools to plot the spectra and the movies are included. Specific scripts such as `test_color.py`, `test_speed.py`, `test_radial.py` and `test_orientation.py` explore the role of different parameters for each individual envelope (respectively color, speed, radial frequency, orientation). Our aim is to keep the code as simple as possible in order to be comprehensible and flexible. To sum up, when we build a custom  Motion Cloud there are 3 simple steps to follow:

1. set the MC parameters and construct the Fourier envelope, then visualize it as iso-surfaces:

```python
import MotionClouds as mc
import numpy as np
# define Fourier domain
fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
# define an envelope
envelope = mc.envelope_gabor(fx, fy, ft,
    V_X=1., V_Y=0., B_V=.1,
    sf_0=.15, B_sf=.1,
    theta=0., B_theta=np.pi/8, alpha=1.)
# Visualize the Fourier Spectrum
mc.visualize(envelope)
```

2. perform the IFFT and contrast normalization; visualize the stimulus as a 'cube' visualization of the image sequence,

```python
movie = mc.random_cloud(envelope)
movie = mc.rectif(movie)
# Visualize the Stimulus
mc.cube(movie, name=name + '_cube')
```

3. export the stimulus as a movie (.mpeg format available), as separate frames (.bmp and .png formats available) in a compressed zipped folder, or as a Matlab matrix (.mat format).

```python
mc.anim_save(movie, name, display=False, vext='.mpeg')
```

If some parameters are not given, they are set to default values corresponding to a ''standard'' Motion Cloud. Moreover, the user can easily explore a range of different Motion Clouds simply by setting  an array of values for a determined parameter. Here, for example, we generate 8 MCs with increasing spatial frequency `sf_0` while keeping the other parameters fixed to default values:

```python
for sf_0 in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    name_ = 'figures/' + name + '-sf_0-' + str(sf_0).replace('.', '_')
    # function performing plots for a given set of parameters
    mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0)
```
***
