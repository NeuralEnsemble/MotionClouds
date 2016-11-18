#!/usr/bin/env python3
# -*- coding: utf8 -*-
from setuptools import setup, find_packages

NAME = "MotionClouds"
import MotionClouds
version = MotionClouds.__version__

setup(
    name = NAME,
    version = version,
    packages = find_packages('src', exclude='docs'),
    package_dir = {'': 'src'},
    py_modules = ['MotionClouds'],
    install_requires=['numpy'],
    extras_require={
                'html' : [
                         'vispy',
                         'matplotlib'
                         'jupyter>=1.0']
    },
    author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "Model-based stimulus synthesis of natural-like random textures for the study of motion perception.",
    long_description=open("README.md").read(),
    license = "GPLv2",
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'parameters'),
    url = 'https://github.com/NeuralEnsemble/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/NeuralEnsemble/' + NAME + '/tarball/' + version,
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                  ],
     )
