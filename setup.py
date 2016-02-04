#!/usr/bin/env python3
# -*- coding: utf8 -*-

from setuptools import setup, find_packages

NAME = "MotionClouds"
version = "0.2"

setup(
    name = NAME,
    version = version,
    packages=find_packages('src', exclude='docs'),
    py_modules = ['MotionClouds'], 
    install_requires=['numpy'],
    package_dir = {'': 'src'},
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
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                  ],
     )
