# /usr/bin/env python
# -*- coding: utf8 -*-
"""

Main script for generating Motion Clouds

(c) Laurent Perrinet - INT/CNRS

Motion Clouds (keyword) parameters:
size    -- power of two to define the frame size (N_X, N_Y)
size_T  -- power of two to define the number of frames (N_frame)
N_X     -- frame size horizontal dimension [px]
N_Y     -- frame size vertical dimension [px]
N_frame -- number of frames [frames] (a full period in time frames)
alpha   -- exponent for the color envelope.
sf_0    -- mean spatial frequency relative to the sampling frequency.
ft_0    -- spatiotemporal scaling factor.
B_sf    -- spatial frequency bandwidth
V_X     -- horizontal speed component
V_Y     -- vertical speed component
B_V     -- speed bandwidth
theta   -- mean orientation of the Gabor kernel
B_theta -- orientation bandwidth
loggabor-- (boolean) if True it uses a log-Gabor kernel (instead of the traditional gabor)

Display parameters:

vext       -- movie format. Stimulus can be saved as a 3D (x-y-t) multimedia file: .mpg or .mp4 movie, .mat array, .zip folder with a frame sequence.
ext        -- frame image format.
T_movie    -- movie duration [s].
fps        -- frame per seconds

"""

import os
DEBUG = False
if DEBUG:
    size = 5
    size_T = 5
    figsize = (400, 400)  # faster
else:
    size = 7
    size_T = 7
    figsize = (600, 600) # nice size, but requires more memory

# TODO: use a parameter file
import numpy as np
N_X = 2**size
N_Y = N_X
N_frame = 2**size_T
ft_0 = np.inf
# default parameters for the "standard Motion Cloud"
alpha = 1.0
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
if not(os.path.isdir(figpath)):os.mkdir(figpath)
recompute = False

def get_grids(N_X, N_Y, N_frame):
    """
        Use that function to define a reference outline for envelopes in Fourier space.
        In general, it is more efficient to define dimensions as powers of 2.
        output is always  of even size..

    """
    fx, fy, ft = np.mgrid[(-N_X//2):((N_X-1)//2 + 1), (-N_Y//2):((N_Y-1)//2 + 1), (-N_frame//2):((N_frame-1)//2 + 1)]
    fx, fy, ft = fx*1./N_X, fy*1./N_Y, ft*1./N_frame
    return fx, fy, ft

def frequency_radius(fx, fy, ft, ft_0=ft_0):
    """
     Returns the frequency radius. To see the effect of the scaling factor run
     'test_color.py'

    """
    N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
    if ft_0==np.inf:
        R2 = fx**2 + fy**2
        R2[N_X//2 , N_Y//2 , :] = np.inf
    else:
        R2 = fx**2 + fy**2 + (ft/ft_0)**2 # cf . Paul Schrater 00
        R2[N_X//2 , N_Y//2 , N_frame//2 ] = np.inf
    return np.sqrt(R2)

def envelope_color(fx, fy, ft, alpha=alpha, ft_0=ft_0):
    """
    Returns the color envelope.
    Run 'test_color.py' to see the effect of alpha
    alpha = 0 white
    alpha = 1 pink
    alpha = 2 red/brownian
    (see http://en.wikipedia.org/wiki/1/f_noise )
    """
    N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
    f_radius = frequency_radius(fx, fy, ft, ft_0=ft_0)**alpha
    if ft_0==np.inf:
        f_radius[N_X//2 , N_Y//2 , : ] = np.inf
    else:
        f_radius[N_X//2 , N_Y//2 , N_frame//2 ] = np.inf
    return 1. / f_radius

def envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, ft_0=ft_0, loggabor=loggabor):
    """
    Radial frequency envelope:
    selects a sphere around a preferred frequency with a shell width B_sf.
    Run 'test_radial.py' to see the explore the effect of sf_0 and B_sf
    """
    if sf_0 == 0.: return 1.
    if loggabor:
        # see http://en.wikipedia.org/wiki/Log-normal_distribution
        fr = frequency_radius(fx, fy, ft, ft_0=ft_0)
        env = 1./fr*np.exp(-.5*(np.log(fr/sf_0)**2)/(np.log((sf_0+B_sf)/sf_0)**2))
        return env
    else:
        return np.exp(-.5*(frequency_radius(fx, fy, ft, ft_0=ft_0) - sf_0)**2/B_sf**2)

def envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V):
    """
     Speed envelope:
     selects the plane corresponding to the speed (V_X, V_Y) with some thickness B_V

     (V_X, V_Y) = (0,1) is downward and  (V_X, V_Y) = (1, 0) is rightward in the movie.
     A speed of V_X=1 corresponds to an average displacement of 1/N_X per frame.
     To achieve one spatial period in one temporal period, you should scale by
     V_scale = N_X/float(N_frame)
     If N_X=N_Y=N_frame and V=1, then it is one spatial period in one temporal
     period. it can be seen in the MC cube. Define ft_0 = N_X/N_frame

    Run 'test_speed.py' to explore the speed parameters

    """
    env = np.exp(-.5*((ft+fx*V_X+fy*V_Y))**2/(B_V*frequency_radius(fx, fy, ft, ft_0=ft_0))**2)
    return env

def envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta):
    """
    Orientation envelope:
    selects one central orientation theta, B_theta the spread
    We use a von-Mises distribution on the orientation.

    Run 'test_orientation.py' to see the effect of changing theta and B_theta.
    """
    if not(B_theta is np.inf):
        angle = np.arctan2(fy, fx)
        envelope_dir = np.exp(np.cos(2*(angle-theta))/B_theta)
        return envelope_dir
    else: # for large bandwidth, returns a strictly flat envelope
        return 1.

def envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha):
    """
    Returns the Motion Cloud kernel

    """
    # TODO : issue a warning if more than 10% of the energy of the envelope falls off the Fourier cube
    # TODO : use a disk mask to ensure all orientations are evely chosen
    envelope = envelope_color(fx, fy, ft, alpha=alpha)
    envelope *= envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta)
    envelope *= envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor)
    envelope *= envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V)
    return envelope

def random_cloud(envelope, seed=None, impulse=False, do_amp=False, threshold=1.e-3):
    """
    Returns a Motion Cloud movie as a 3D matrix.
    It first creates a random phase spectrum and then it computes the inverse FFT to obtain
    the spatiotemporal stimulus.

    - use a specific seed to specify the RNG's seed,
    - test the impulse response of the kernel by setting impulse to True
    - test the effect of randomizing amplitudes too by setting do_amp to True
shape

    """
    (N_X, N_Y, N_frame) = envelope.shape
    amps = 1.
    if impulse:
        phase = 0.
    else:
        np.random.seed(seed=seed)
        phase = 2 * np.pi * np.random.rand(N_X, N_Y, N_frame)
        if do_amp:
            # see Galerne, B., Gousseau, Y. & Morel, J.-M. Random phase textures: Theory and synthesis. IEEE Transactions in Image Processing (2010). URL http://www.biomedsearch.com/nih/Random-Phase-Textures-Theory-Synthesis/20550995.html. (basically, they conclude "Even though the two processes ADSN and RPN have different Fourier modulus distributions (see Section 4), they produce visually similar results when applied to natural images as shown by Fig. 11.")
            amps = np.random.randn(N_X, N_Y, N_frame)

    Fz = amps * envelope * np.exp(1j * phase)

    # centering the spectrum
    Fz = np.fft.ifftshift(Fz)
    Fz[0, 0, 0] = 0.
    z = np.fft.ifftn((Fz)).real
    return z

########################## Display Tools #######################################
vext = '.webm'
ext = '.png'
T_movie = 8. # this value defines the duration of a temporal period

# display parameters
try:
    import pyprind as progressbar
    PROGRESS = True
except:
    PROGRESS = False

os.environ['ETS_TOOLKIT'] = 'qt4' # Works in Mac
# os.environ['ETS_TOOLKIT'] = 'wx' # Works in Debian
MAYAVI = 'Import'
#MAYAVI = 'Avoid' # uncomment to avoid generating mayavi visualizations (and save some memory...)
def import_mayavi():
    global MAYAVI, mlab
    if (MAYAVI == 'Import'):
        try:
            from mayavi import mlab
            MAYAVI = 'Ok : New and shiny'
            print('Imported Mayavi')
        except:
            try:
                from enthought.mayavi import mlab
                print('Seems you have an old implementation of MayaVi, but things should work')
                MAYAVI = 'Ok but old'
                print('Imported Mayavi')
            except:
               print('Could not import Mayavi')
               MAYAVI = 'Could not import Mayavi'
    elif (MAYAVI == 'Could not import Mayavi') or (MAYAVI == 'Ok : New and shiny') or (MAYAVI == 'Ok but old'):
        pass # no need to import that again
    else:
        print('We have chosen not to import Mayavi')
# Trick from http://github.enthought.com/mayavi/mayavi/tips.html : to use offscreen rendering, try xvfb :1 -screen 0 1280x1024x24 in one terminal, export DISPLAY=:1 before you run your script

def get_size(mat):
    """
    Get stimulus dimensions

    """
    return [np.size(mat, axis=k) for k in range(np.ndim(mat))]

#NOTE: Python uses the first dimension (rows) as vertical axis and this is the Y in the spatiotemporal domain. Be careful with the convention of X and Y.

def visualize(z, azimuth=290., elevation=45.,
    thresholds=[0.94, .89, .75, .5, .25, .1], opacities=[.9, .8, .7, .5, .2, .1],
    name=None, ext=ext, do_axis=True, do_grids=False, draw_projections=True,
    colorbar=False, f_N=2., f_tN=2., figsize=figsize):

    """ Visualize the  Fourier spectrum by showing 3D contour plots at different thresholds
    
    parameters
    ----------
    z : envelope of the cloud

    """
    import_mayavi()

    N_X, N_Y, N_frame = z.shape
    fx, fy, ft = get_grids(N_X, N_Y, N_frame)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=figsize)
    mlab.clf()

    # Normalize the amplitude.
    z /= z.max()
    # Create scalar field
    src = mlab.pipeline.scalar_field(fx, fy, ft, z)
    if draw_projections:
        src_x = mlab.pipeline.scalar_field(fx, fy, ft, np.tile(np.sum(z, axis=0), (N_X, 1, 1)))
        src_y = mlab.pipeline.scalar_field(fx, fy, ft, np.tile(np.reshape(np.sum(z, axis=1), (N_X, 1, N_frame)), (1, N_Y, 1)))
        src_z = mlab.pipeline.scalar_field(fx, fy, ft, np.tile(np.reshape(np.sum(z, axis=2), (N_X, N_Y, 1)), (1, 1, N_frame)))

        # Create projections
        border = 0.47
        scpx = mlab.pipeline.scalar_cut_plane(src_x, plane_orientation='x_axes', view_controls=False)
        scpx.implicit_plane.plane.origin = [-border, 1/N_Y, 1/N_frame]
        scpx.enable_contours = True
        scpy = mlab.pipeline.scalar_cut_plane(src_y, plane_orientation='y_axes', view_controls=False)
        scpy.implicit_plane.plane.origin = [1/N_X, border, 1/N_frame]
        scpy.enable_contours = True
        scpz = mlab.pipeline.scalar_cut_plane(src_z, plane_orientation='z_axes', view_controls=False)
        scpz.implicit_plane.plane.origin = [1/N_X, 1/N_Y, -border]
        scpz.enable_contours = True

    # Generate iso-surfaces at different energy levels
    for threshold, opacity in zip(thresholds, opacities):
        mlab.pipeline.iso_surface(src, contours=[z.max()-threshold*z.ptp(), ], opacity=opacity)
        mlab.outline(extent=[-1./2, 1./2, -1./2, 1./2, -1./2, 1./2],)

    # Draw a sphere at the origin
    x = np.array([0])
    y = np.array([0])
    z = np.array([0])
    s = 0.01
    mlab.points3d(x, y, z, extent=[-s, s, -s, s, -s, s], scale_factor=0.15)

    if colorbar: mlab.colorbar(title='density', orientation='horizontal')
    if do_axis:
        ax = mlab.axes(xlabel='fx', ylabel='fy', zlabel='ft',)
        ax.axes.set(font_factor=2.)

    try:
        mlab.view(azimuth=azimuth, elevation=elevation, distance='auto', focalpoint='auto')
    except:
        print(" You should upgrade your mayavi version")

    if not(name is None):
        mlab.savefig(name + ext, magnification='auto', size=figsize)
    else:
       mlab.show(stop=True)

    mlab.close(all=True)

def cube(im, azimuth=-45., elevation=130., roll=-180., name=None,
         ext=ext, do_axis=True, show_label=True, colormap='gray',
         vmin=0., vmax=1., figsize=figsize):

    """
    Visualize the stimulus as a cube

    """
    import_mayavi()

    N_X, N_Y, N_frame = im.shape
    fx, fy, ft = get_grids(N_X, N_Y, N_frame)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=figsize)
    mlab.clf()
    src = mlab.pipeline.scalar_field(fx*2., fy*2., ft*2., im)

    mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', slice_index=0,
                                     colormap=colormap, vmin=vmin, vmax=vmax)
    mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', slice_index=N_frame, 
                                     colormap=colormap, vmin=vmin, vmax=vmax)
    mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', slice_index=0,
                                     colormap=colormap, vmin=vmin, vmax=vmax)
    mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', slice_index=N_X,
                                     colormap=colormap, vmin=vmin, vmax=vmax)
    mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', slice_index=0,
                                     colormap=colormap, vmin=vmin, vmax=vmax)
    mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', slice_index=N_Y,
                                     colormap=colormap, vmin=vmin, vmax=vmax)

    if do_axis:
        ax = mlab.axes(xlabel='x', ylabel='y', zlabel='t',
                       ranges=[0., N_X, 0., N_Y, 0., N_frame],
                       x_axis_visibility=False, y_axis_visibility=False,
                       z_axis_visibility=False)
        ax.axes.set(font_factor=2.)

        if not(show_label): ax.axes.set(label_format='')


    try:
        mlab.view(azimuth=azimuth, elevation=elevation, distance='auto', focalpoint='auto')
        mlab.roll(roll=roll)
    except:
        print(" You should upgrade your mayavi version")

    if not(name is None):
        mlab.savefig(name + ext, magnification='auto', size=figsize)
    else:
        mlab.show(stop=True)

    mlab.close(all=True)

def check_if_anim_exist(filename, vext=vext):
    """
    Check if the movie already exists

    returns True if the movie does not exist, False if it does

    """
    return not(os.path.isfile(os.path.join(figpath, filename + vext)))

SUPPORTED_FORMATS = ['.h5', '.mpg', '.mp4', '.gif', '.webm', '.zip', '.mat', '.mkv']
def anim_save(z, filename, display=True, flip=False, vext=vext,
              centered=False, T_movie=T_movie, verbose=True):
    """
    Saves a numpy 3D matrix (x-y-t) to a multimedia file.

    The input pixel values are supposed to lie in the [0, 1.] range.

    """
    import os                         # For issuing commands to the OS.
    import tempfile
    from scipy.misc.pilutil import toimage
    fps = int(z.shape[-1] / T_movie)
    def make_frames(z):
        N_X, N_Y, N_frame = z.shape
        files = []
        tmpdir = tempfile.mkdtemp()

        if PROGRESS:
            pbar = progressbar.ProgPercent(N_frame, monitor=True)
        print('Saving sequence ' + filename + vext)
        for frame in range(N_frame):
            if PROGRESS: pbar.update()
            fname = os.path.join(tmpdir, 'frame%03d.png' % frame)
            image = np.rot90(z[:, :, frame])
            if flip: image = np.flipud(image)
            toimage(image, high=255, low=0, cmin=0., cmax=1., pal=None,
                    mode=None, channel_axis=None).save(fname)
            files.append(fname)

        if PROGRESS: print(pbar)
        return tmpdir, files

    def remove_frames(tmpdir, files):
        """
        Remove frames from the temp folder

        """
        for fname in files: os.remove(fname)
        if not(tmpdir == None): os.rmdir(tmpdir)

    if verbose:
        verb_ = ''
    else:
        verb_ = ' 2>/dev/null'
    if vext == '.mpg':
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
#        cmd = 'ffmpeg -v 0 -y -sameq -loop_output 0 -r ' + str(fps) + ' -i ' + tmpdir + '/frame%03d.png  ' + filename + vext # + ' 2>/dev/null')
        #cmd = 'ffmpeg -v 0 -y -sameq  -loop_output 0 -i ' + tmpdir + '/frame%03d.png  ' + filename + vext # + ' 2>/dev/null')
        options = ' -f image2  -r ' + str(fps) + ' -y '
        os.system('ffmpeg -i ' + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_)
        # 3) clean up
        #remove_frames(tmpdir, files)
    if vext == '.mp4': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
#         options = ' -y -f image2pipe -c:v png -i - -c:v libx264 -preset ultrafast -qp 0 -movflags +faststart -pix_fmt yuv420p '
#         options += ' -g ' + str(fps) + '  -r ' + str(fps) + ' '
#         cmd = 'cat '  + tmpdir + '/*.png  | ffmpeg '  + options + filename + vext + verb_
        options = ' -f mp4 -pix_fmt yuv420p -c:v libx264  -g ' + str(fps) + '  -r ' + str(fps) + ' '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    if vext == '.webm':
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        options = ' -f webm  -pix_fmt yuv420p -vcodec libvpx -qmax 12 -g ' + str(fps) + '  -r ' + str(fps) + ' '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    if vext == '.mkv': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        options = ' -y -f image2pipe -c:v png -i - -c:v libx264 -preset ultrafast -qp 0 -movflags +faststart -pix_fmt yuv420p  -g ' + str(fps) + '  -r ' + str(fps) + ' '
        cmd = 'cat '  + tmpdir + '/*.png  | ffmpeg '  + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    if vext == '.gif': # http://www.uoregon.edu/~noeckel/MakeMovie.html
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
#        options = ' -pix_fmt rgb24 -r ' + str(fps) + ' -loop_output 0 '
#        os.system('ffmpeg -i '  + tmpdir + '/frame%03d.png  ' + options + filename + vext + ' 2>/dev/null')
        options = ' -set delay 8 -colorspace GRAY -colors 256 -dispose 1 -loop 0 '
        os.system('convert '  + tmpdir + '/frame*.png  ' + options + filename + vext + verb_)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.png':
        toimage(np.flipud(z[:, :, 0]).T, high=255, low=0, cmin=0., cmax=1., pal=None, mode=None, channel_axis=None).save(filename + vext)

    elif vext == '.zip':
        # TODO : give the possiblity to specify the format of files inside the zip
        tmpdir, files = make_frames(z)
        import zipfile
        zf = zipfile.ZipFile(filename + vext, "w")
        # convert to BMP for optical imaging
        files_bmp = []
        for fname in files:
            fname_bmp = os.path.splitext(fname)[0] + '.bmp'
            # print fname_bmp
            os.system('convert ' + fname + ' ppm:- | convert -size 256x256+0 -colors 256 -colorspace Gray - BMP2:' + fname_bmp) # to generate 8-bit bmp (old format)
            files_bmp.append(fname_bmp)
            zf.write(fname_bmp)
        zf.close()
        remove_frames(tmpdir=None, files=files_bmp)
        remove_frames(tmpdir, files)

    elif vext == '.mat':
        from scipy.io import savemat
        savemat(filename + vext, {'z':z})

    elif vext == '.h5':
        from tables import openFile, Float32Atom
        hf = openFile(filename + vext, 'w')
        o = hf.createCArray(hf.root, 'stimulus', Float32Atom(), z.shape)
        o = z
        #   print o.shape
        hf.close()

def play(z, T=5.):
    """
    T: duration in second of a period

    TODO: currently failing on MacOsX - use pyglet?

    """
    global t, t0, frames
    N_X, N_Y, N_frame = z.shape
    import glumpy
    fig = glumpy.figure((N_X, N_Y))
    Z = z[:, :, 0].T.astype(np.float32)
    image = glumpy.image.Image(Z) #, interpolation='nearest', colormap=glumpy.colormap.Grey, vmin=0, vmax=1)
    t0, frames, t = 0, 0, 0

    @fig.event
    def on_draw():
        fig.clear()
        image.draw(x=0, y=0, z=0, width=fig.width, height=fig.height )
    @fig.event
    def on_key_press(symbol, modifiers):
        if symbol == glumpy.window.key.TAB:
            if fig.window.get_fullscreen():
                fig.window.set_fullscreen(0)
            else:
                fig.window.set_fullscreen(1)
        if symbol == glumpy.window.key.ESCAPE:
            import sys
            sys.exit()

    @fig.event
    def on_idle(dt):
        global t, t0, frames
        t += dt
        frames = frames + 1
        if t-t0 > 5.0:
            fps = float(frames)/(t-t0)
            print 'FPS: %.2f (%d frames in %.2f seconds)' % (fps, frames, t-t0)
            frames, t0 = 0, t
         # computing the frame more closely to the actual time
        Z[...] = z[:, :, np.int(np.mod(t, T)/T * N_frame)].T.astype(np.float32)
        #Z[...] = z[:, :, frames % N_frame].T.astype(np.float32)
        image.update()
        fig.redraw()
    glumpy.show()

def rectif(z, contrast=.9, method='Michelson', verbose=False):
    """
    Transforms an image (can be 1,2 or 3D) with normal histogram into
    a 0.5 centered image of determined contrast
    method is either 'Michelson' or 'Energy'

    """
    # Phase randomization takes any image and turns it into Gaussian-distributed noise of the same power (or, equivalently, variance).
    # See: Peter J. Bex J. Opt. Soc. Am. A/Vol. 19, No. 6/June 2002 Spatial frequency, phase, and the contrast of natural images

    # Final rectification
    if verbose:
        print('Before Rectification of the frames')
        print( 'Mean=', np.mean(z[:]), ', std=', np.std(z[:]), ', Min=', np.min(z[:]), ', Max=', np.max(z[:]), ' Abs(Max)=', np.max(np.abs(z[:])))

    z -= np.mean(z[:]) # this should be true *on average* in MotionClouds

    if (method == 'Michelson'):
        z = (.5* z/np.max(np.abs(z[:]))* contrast + .5)
    else:
        z = (.5* z/np.std(z[:])  * contrast + .5)

    if verbose:
        import pylab
        pylab.hist(z.ravel())

        print('After Rectification of the frames')
        print('Mean=', np.mean(z[:]), ', std=', np.std(z[:]), ', Min=', np.min(z[:]), ', Max=', np.max(z[:]))
        print('percentage pixels clipped=', np.sum(np.abs(z[:])>1.)*100/z.size)
    return z

def figures_MC(fx, fy, ft, name, V_X=V_X, V_Y=V_Y, do_figs=True, do_movie=True,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha, vext=vext,
                    seed=None, impulse=False, do_amp=False, verbose=False):
    """
    Generates the figures corresponding to the Fourier spectra and the stimulus cubes and
    movies.
    The figures names are automatically generated.
    """
    if check_if_anim_exist(name, vext=vext):
        z = envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha)
        figures(z, name, vext=vext, do_figs=do_figs, do_movie=do_movie,
                    seed=seed, impulse=impulse, verbose=verbose, do_amp=do_amp)
    else:
        figures(z=None, name=name, vext=vext, do_figs=do_figs, do_movie=do_movie,
                    seed=seed, impulse=impulse, verbose=verbose, do_amp=do_amp)

def figures(z=None, name='MC', vext=vext, do_movie=True, do_figs=True,
                    seed=None, impulse=False, verbose=False, masking=False, do_amp=False):

    import_mayavi()

    if ((MAYAVI == 'Import') or MAYAVI[:2]=='Ok') and do_figs and check_if_anim_exist(name, vext=ext):
        visualize(z, name=os.path.join(figpath, name))           # Visualize the Fourier Spectrum

    if (do_movie and check_if_anim_exist(name, vext=vext)) or (((MAYAVI == 'Import') or MAYAVI[:2]=='Ok') and do_figs and check_if_anim_exist(name + '_cube', vext=ext)):
        movie = rectif(random_cloud(z, seed=seed, impulse=impulse, do_amp=do_amp), verbose=verbose)

    if (((MAYAVI == 'Import') or MAYAVI[:2]=='Ok') and do_figs and check_if_anim_exist(name + '_cube', vext=ext)):
        cube(movie, name=os.path.join(figpath, name + '_cube'))   # Visualize the Stimulus cube

    if (do_movie) and check_if_anim_exist(name, vext=vext):
        anim_save(movie, os.path.join(figpath, name), display=False, vext=vext)

    if notebook:
        in_show_video(name)

def in_show_video(name, loop=True):
    import os
    from IPython.core.display import display, Image, HTML
    from base64 import b64encode
    try: #if MAYAVI[:2]=='Ok':
        with open(os.path.join(figpath, name + ext), "r") as image_file:
            im1 = 'data:image/png;base64,' + b64encode(image_file.read())
        with open(os.path.join(figpath, name + '_cube' + ext), "r") as image_file:
            im2 = 'data:image/png;base64,' + b64encode(image_file.read())
        with open(os.path.join(figpath, name + vext), "r") as video_file:
            im3 = 'data:video/webm;base64,' + b64encode(video_file.read())
        s = """
        <center><table border=none width=100%% height=100%%>
        <tr>
        <td width=33%%><center><img src="%s" width=100%%/></td>
        <td rowspan=2  colspan=2><center><video src="%s" autoplay="autoplay" loop="loop" type="video/%s" width=100%%/></td>
        </tr>
        <tr>
        <td><center><img src="%s" width=100%%/></td>
        </tr>
        </table></center>"""%(im1, im3, vext, im2)
        t=HTML(s)
        print name
        display(t)
    except: #else:
        video = open(os.path.join(figpath, name + vext), "rb").read()
        video_encoded = b64encode(video)
        if vext=='.webm':
            video_tag = '<video controls  autoplay="autoplay" loop="loop" width=50% src="data:video/webm;base64,{0}">'.format(video_encoded)
        else:
            video_tag = '<video controls  autoplay="autoplay" loop="loop" width=50% src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
        display(HTML(data=video_tag))

