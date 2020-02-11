from setuptools import setup, find_packages

NAME = "MotionClouds"
import MotionClouds
VERSION = MotionClouds.__version__ # << to change in __init__.py

setup(
    name = NAME,
    version = VERSION,
    packages = find_packages('MotionClouds', exclude=['docs', 'figures', 'website', 'dist']),
    package_dir = {'': NAME},
    py_modules = [NAME],
    install_requires=['numpy'],
    extras_require={
                'html' : [
                         'vispy',
                         'matplotlib'
                         'jupyter']
    },
    author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "Model-based stimulus synthesis of natural-like random textures for the study of motion perception.",
    long_description="""
MotionClouds
============

**MotionClouds** are random dynamic stimuli optimized to study motion perception.

See https://github.com/NeuralEnsemble/MotionClouds for more information.

    """,
    license = "GPLv2",
    keywords = ['computational neuroscience', 'simulation', 'analysis', 'visualization', 'parameters'],
    url = 'https://github.com/NeuralEnsemble/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/NeuralEnsemble/' + NAME + '/tarball/' + VERSION,
    classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Medical Science Apps.',
                   'Topic :: Scientific/Engineering :: Image Recognition',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7',
                  ],
     )
