#!/usr/bin/env python
# -*- coding: utf8 -*-

# from distutils.core import setup
from setuptools import setup

NAME = "MotionClouds"
version = "0.2"

setup(
    name = NAME,
    version = version,
    packages = [NAME],
    package_dir = {NAME: NAME},
    author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "Model-based stimulus synthesis of natural-like random textures for the study of motion perception.",
#     long_description=open("README.md").read(),
    license = "GPLv2",
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'parameters'),
    url = 'https://github.com/NeuralEnsemble/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/NeuralEnsemble/' + NAME + '/tarball/' + version,
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                  ],
     )
