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

# sudo aptitude install python-pip python-numpy python-scipy mayavi2 python-matplotlib ffmpeg
# sudo aptitude install texlive-latex-recommended  latexmk latexdiff

# 3) full install with python editor and libraries for various export types

sudo aptitude install python-pip python-numpy mayavi2 python-matplotlib spyder python-tables imagemagick texlive-latex-recommended latexmk latexdiff zip ipython psychopy
# http://askubuntu.com/questions/432542/is-ffmpeg-missing-from-the-official-repositories-in-14-04
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo aptitude install ffmpeg

pip install --user pyprind

