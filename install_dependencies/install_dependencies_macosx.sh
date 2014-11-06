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
echo "export PATH=/usr/local/bin:/usr/local/sbin:$HOME/.gem/ruby/1.8/bin:$PATH" >> ~/.bash_profile
echo "export PYTHONPATH=/usr/local/lib/python2.7/site-packages:/usr/local/opt/vtk5/lib/python2.7/site-packages:$PYTHONPATH" >> ~/.bash_profile
echo "export QT_API=pyqt" >> ~/.bash_profile
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

# mayavi
# http://davematthew.blogspot.fr/2013/10/installing-matplotlib-and-mayavi-on-mac.html
brew install cmake
brew install qt
brew install pyqt

#brew tap iMichka/homebrew-MacVTKITKPythonBottles
#brew install iMichka/MacVTKITKPythonBottles/imichka-vtk --with-qt --with-matplotlib --with-python

brew install vtk5 --with-qt
#ln -s /usr/local/opt/vtk5/lib/python2.7/site-packages/vtk/ /usr/local/lib/python2.7/site-packages/vtk # there is a bug in the onstallation of vtk such that it can not be imported
#brew install vtk --python
pip install -U git+https://github.com/enthought/traitsgui
pip install -U git+https://github.com/enthought/traitsbackendqt
pip install -U configobj
pip install -U envisage
# pip install "Mayavi[app]"
pip install -U git+https://github.com/enthought/mayavi

# HDF export
brew install hdf5
pip install cython==0.13 #tables doesn't build with last upgrade
pip install -U numexpr
pip install -U tables

# brew uninstall wxpython
# pip install -U psychopy

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
