# MotionClouds

**MotionClouds** are random dynamic stimuli optimized to study motion perception.

In particular, these stimuli can be made closer to naturalistic textures compared to usual stimuli such as gratings and random-dot kinetograms. These have controlled information content: We simplified the definition to parametrically define these "Motion Clouds" around the most prevalent feature axis (mean and bandwith): direction, scale (spatial frequency), orientation. These scripts implement a framework to generate these random texture movies. 

<center><img src="http://www.motionclouds.invibe.net/files/MotionPlaid_comp1.gif" width="50%"></center>


* to install the package, run:
````
pip install MotionClouds
````

* to install the dependencies on a debian-like system:
````
# installing dependencies on Debian for MotionClouds
# --------------------------------------------------
sudo apt-get install git aptitude
# it is always a good idea to update/upgrade your system before
sudo aptitude update
sudo aptitude upgrade

# A script for the impatient:
# uncomment to fit your installation preference
# others should read the README.txt doc.

# 1) minimal install

# sudo aptitude install python-numpy

# 2) minimal install with visualization and generation of documentation

# sudo aptitude install python-pip python-numpy python-scipy python-matplotlib ffmpeg
# sudo aptitude install texlive-latex-recommended  latexmk latexdiff

# 3) full install with python editor and libraries for various export types

sudo aptitude install python-pip python-numpy python-matplotlib spyder python-tables imagemagick texlive-latex-recommended latexmk latexdiff zip ipython psychopy
# http://askubuntu.com/questions/432542/is-ffmpeg-missing-from-the-official-repositories-in-14-04
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo aptitude install ffmpeg

pip install --user pyprind
# vispy
pîp install --user pyglet
pip install --user git+https://github.com/vispy/vispy
````

* to install the dependencies on a MacOsX system:
````
#!/bin/bash

# 1. install Xcode command-line tools:
# From this url : (uncomment the following line)
# open http://itunes.apple.com/us/app/xcode/id497799835?mt=12
# install Xcode on the Mac App Store by clicking on “View in Mac App Store”.
# or simpler, issue
git
# a pop-up window should appear which recommends to install the command-line tools.

# on MacOsX Yosemite, the following also works:
# xcode-select --install

# 2. install HomeBrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# to reinstall, do:
# rm -rf /usr/local/Cellar /usr/local/.git && brew cleanup

# adding correct variables
# (see http://stackoverflow.com/questions/26775145/adding-directory-to-a-path-variable-like-path-just-once-in-bash )
var=$(echo $PATH | grep -o  '/usr/local/bin')
if [ -n "$var" ] ; then
	echo "export PATH=/usr/local/bin:$PATH" >> ~/.bash_profile
fi
var=$(echo $PATH | grep -o  '/usr/local/sbin')
if [ -n "$var" ] ; then
	echo "export PATH=/usr/local/sbin:$PATH" >> ~/.bash_profile
fi
var=$(echo $PYTHONPATH | grep -o  '/usr/local/lib/python2.7/site-packages')
if [ -n "$var" ] ; then
	echo "export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH" >> ~/.bash_profile
fi
var=$(echo $PYTHONPATH | grep -o  '/usr/local/opt/vtk5/lib/python2.7/site-packages')
if [ -n "$var" ] ; then
	echo "export PYTHONPATH=/usr/local/opt/vtk5/lib/python2.7/site-packages:$PYTHONPATH" >> ~/.bash_profile
fi
var=$(echo $QT_API | grep -o  'pyqt')
if [ -n "$var" ] ; then
	echo "export QT_API=pyqt" >> ~/.bash_profile
fi
source ~/.bash_profile
# Make sure we’re using the latest Homebrew
brew install git
brew update
# Upgrade any already-installed formulae
brew upgrade

# brew uninstall python pyqt pyside vtk
# brew uninstall wxpython
# mv /usr/local/lib/python2.7/site-packages /usr/local/lib/python2.7/site-packages-old 
# mv /usr/local/share/python /usr/local/share/python-old
# install python through HomeBrew
brew install python --framework --universal

# bootstrap pip
pip install --upgrade setuptools
pip install --upgrade distribute

# installing xquartz
brew tap caskroom/cask
brew install brew-cask
brew cask install xquartz

# numpy et al
brew tap homebrew/science
brew tap Homebrew/python
brew install gcc
brew install fftw
brew install libtool
brew install numpy #--with-openblas
brew test numpy
brew install scipy
brew install pillow
pip install -U pandas
pip install -U nose
pip install -U ipython

# pylab
brew install matplotlib --with-tex

# vispy
pîp install -U pyglet
pip install -U git+https://github.com/vispy/vispy

# HDF export
brew install hdf5
pip install cython==0.13 #tables doesn't build with last upgrade
pip install -U numexpr
pip install -U tables

# brew uninstall wxpython
pip install -U psychopy

pip install -U psutil
pip install -U pyprind


# install online displaying tools
# pip install PyOpenGL PyOpenGL_accelerate
# pip install glumpy
# brew install --HEAD smpeg
# brew install pygame
# brew install mercurial
# pip install hg+https://pyglet.googlecode.com/hg/
pip install -U NeuroTools
pip install -U git+https://github.com/NeuralEnsemble/MotionClouds

# convert
brew install imagemagick
brew install x264
brew install ffmpeg --with-libvpx

# Remove outdated versions from the cellar
brew cleanup
python -c 'import MotionClouds as mc; fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame); z = mc.envelope_gabor(fx, fy, ft); mc.figures(z, "test")'

````

* to install the latest version, use:
````
pip install git+https://github.com/NeuralEnsemble/MotionClouds.git
````


* for more documentation, visit the MotionClouds website:
http://motionclouds.invibe.net/




The description of this method was published in:

* Paula S. Leon, Ivo Vanzetta, Guillaume S. Masson, Laurent U. Perrinet. _Motion Clouds: Model-based stimulus synthesis of natural-like random textures for the study of motion perception._ [**Journal of Neurophysiology**](http://jn.physiology.org/content/early/2012/03/10/jn.00737.2011), 107(11):3217--3226, 2012  [URL](http://invibe.net/LaurentPerrinet/Publications/Sanz12) - [preprint](http://www.motionclouds.invibe.net/files/MotionClouds.pdf) - [Supplementary Information](http://www.motionclouds.invibe.net/files/MotionClouds_Supplementary.pdf) - [Supplementary Videos](http://www.motionclouds.invibe.net/files/MotionClouds_VideoFigures.pdf)

While this method was used in the following paper:

* Claudio Simoncini, Laurent U. Perrinet, Anna Montagnini, Pascal Mamassian, Guillaume S. Masson. _More is not always better: dissociation between perception and action explained by adaptive gain control._ [**Nature Neuroscience**](http://www.nature.com/neuro/journal/v15/n11/full/nn.3229.html), 2012 [URL](http://invibe.net/LaurentPerrinet/Publications/Simoncini12)

This work is supported by the European Union project Number FP7-269921, ``BrainScaleS'' (Brain-inspired multiscale computation in neuromorphic hybrid systems), an EU FET-Proactive FP7 funded research project. The project started on 1 January 2011. It is a collaboration of 18 research groups from 10 European countries.

<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/e/e2/Public--BrainScalesLogo.svg/100px-Public--BrainScalesLogo.svg.png" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/8/88/Public--FET--FETTreeLogo.jpg/70px-Public--FET--FETTreeLogo.jpg" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/3/3b/Public--EU-FP7Logo.gif/90px-Public--EU-FP7Logo.gif" width="10%">
<img src="https://brainscales.kip.uni-heidelberg.de/images/thumb/5/5b/Public--EU-Logo.gif/90px-Public--EU-Logo.gif" width="10%">

***
## Code example - [demo](http://motionclouds.invibe.net/posts/testing_components.html)

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
