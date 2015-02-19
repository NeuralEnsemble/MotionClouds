# TODO: use a parameter file
import numpy as np

PROGRESS = False

size = 7
size_T = 7
figsize = (600, 600) # nice size, but requires more memory

N_X = 2**size
N_Y = N_X
N_frame = 2**size_T

# default parameters for the "standard Motion Cloud"
alpha = 0.0
ft_0 = np.inf
sf_0 = 0.15
B_sf = 0.1
V_X = 1.
V_Y = 0.
B_V = .2
theta = 0.
B_theta = np.pi/32.
loggabor = True

notebook = False
figpath = 'results/'

vext = '.webm'
ext = '.png'
T_movie = 8. # this value defines the duration in seconds of a temporal period
SUPPORTED_FORMATS = ['.h5', '.mpg', '.mp4', '.gif', '.webm', '.zip', '.mat']#, '.mkv']

MAYAVI = 'Import'
#MAYAVI = 'Avoid' # uncomment to avoid generating mayavi visualizations (and save some memory...)
