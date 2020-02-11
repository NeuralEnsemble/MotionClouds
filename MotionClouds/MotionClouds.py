__author__ = "Laurent Perrinet INT - CNRS"
__licence__ = 'GPLv2'
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
import numpy as np


# size of the stimulus
size = 8
size_T = 8
N_X = 2**size
N_Y = N_X
N_frame = 2**size_T

# default parameters for the "standard Motion Cloud"
sf_0 = 0.125
B_sf = 0.1
V_X = 1.
V_Y = 0.
B_V = .5
theta = 0.
B_theta = np.pi/16.

# for backward compatibilty
alpha = 0.0
ft_0 = np.inf
loggabor = True

contrast = 1.
method = 'Michelson'

recompute = False
notebook = False
figpath = '../files/'
figsize = (800, 800) # nice size, but requires more memory

vext = '.mp4'
ext = '.png'
T_movie = N_frame/100. # this value defines the duration in seconds of a temporal period (assuming a 100Hz refresh rate)
SUPPORTED_FORMATS = ['.h5', '.mpg', '.mp4', '.gif', '.webm', '.zip', '.mat', '.png']

def rectif(z_in, contrast=contrast, method=method, verbose=False):
    """
    Transforms an image (can be 1, 2 or 3D) with normal histogram into
    a 0.5 centered image of determined contrast
    method is either 'Michelson' or 'Energy'

    Phase randomization takes any image and turns it into Gaussian-distributed
    noise of the same power (or, equivalently, variance).
    # See: Peter J. Bex J. Opt. Soc. Am. A/Vol. 19, No. 6/June 2002 Spatial
    frequency, phase, and the contrast of natural images
    """
    z = z_in.copy()
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
        print('After Rectification of the frames')
        print('Mean=', np.mean(z[:]), ', std=', np.std(z[:]), ', Min=', np.min(z[:]), ', Max=', np.max(z[:]))
        print('percentage pixels clipped=', np.sum(np.abs(z[:])>1.)*100/z.size)
    return z

def get_grids(N_X, N_Y, N_frame):
    """
        Use that function to define a reference outline for envelopes in Fourier space.
        In general, it is more efficient to define dimensions as powers of 2.
        output is always  of even size.

        A special case is when ``N_frame`` is one, in which case we generate a single frame.

    """
    if N_frame>1:
        fx, fy, ft = np.mgrid[(-N_X//2):((N_X-1)//2 + 1), (-N_Y//2):((N_Y-1)//2 + 1), (-N_frame//2):((N_frame-1)//2 + 1)]
    else:
        fx, fy, ft = np.mgrid[(-N_X//2):((N_X-1)//2 + 1), (-N_Y//2):((N_Y-1)//2 + 1), 0:1]
    fx, fy, ft = fx*1./N_X, fy*1./N_Y, ft*1./N_frame

    return fx, fy, ft

def frequency_radius(fx, fy, ft, ft_0=ft_0, clean_division=False):
    """
     Returns the frequency radius. To see the effect of the scaling factor run
     'test_color.py'

    """
    if ft_0==np.inf:
        f_radius2 = fx**2 + fy**2
    else:
        f_radius2 = fx**2 + fy**2 + (ft/ft_0)**2 # cf . Paul Schrater 00

    if clean_division:
        f_radius2[f_radius2==0.] = np.inf

    return np.sqrt(f_radius2)

def retina(fx, fy, ft, df=.07, sigma=.5):
    """
    A parametric description of the envelope of retinal processsing.
    See https://laurentperrinet.github.io/sciblog/posts/2015-05-21-a-simple-pre-processing-filter-for-image-processing.html
    for more information.

    In digital images, some of the energy in Fourier space is concentrated outside the
    disk corresponding to the Nyquist frequency. Let's design a filter with:

        - a sharp cut-off for radial frequencies higher than the Nyquist frequency,
        - times a smooth but sharp transition (implemented with a decaying exponential),
        - times a high-pass filter designed by one minus a gaussian blur.

    This filter is rotation invariant.

    Note that this filter is defined by two parameters:
        - one for scaling the smoothness of the transition in the high-frequency range,
        - one for the characteristic length of the high-pass filter.

    The first is defined relative to the Nyquist frequency (in absolute values) while the second
    is relative to the size of the image in pixels and is given in number of pixels.
    """
    # removing high frequencies in the corners
    fr = frequency_radius(fx, fy, ft, ft_0=ft_0)
    env = (1-np.exp((fr-.5)/(.5*df)))*(fr<.5)
    # removing low frequencies
    # N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
    # env *= 1-np.exp(-.5*(fr**2)/((sigma/N_X)**2))
    return env

def envelope_color(fx, fy, ft, alpha=alpha, ft_0=ft_0):
    """
    Returns the color envelope.

    In some circonstances, it is desirable to create a texture with a different "color"
    than that of natural images (that is where the envelope is in 1/f).

    Run 'test_color' notebook to see the effect of alpha
    alpha = 0 white
    alpha = 1 pink
    alpha = 2 red/brownian
    (see http://en.wikipedia.org/wiki/1/f_noise )
    """
    if alpha == 0.0:
        return np.ones_like(fx) #np.ones_like(fx)
    else:
        f_radius = frequency_radius(fx, fy, ft, ft_0=ft_0, clean_division=True)
        return f_radius**(-alpha)

def envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, ft_0=ft_0, loggabor=loggabor):
    """
    Returns the radial frequency envelope:

    Selects a preferred spatial frequency ``sf_0`` and a bandwidth ``B_sf``.

    Run the 'test_radial' notebook to see the explore the effect of ``sf_0`` and ``B_sf``, see
    https://neuralensemble.github.io/MotionClouds/posts/testing-radial.html

    """
    if sf_0 == 0. or B_sf==np.inf:
        if loggabor:
            return envelope_color(fx, fy, ft, alpha=1., ft_0=ft_0)
        else:
            return np.ones_like(fx)
    elif loggabor:
        # see http://en.wikipedia.org/wiki/Log-normal_distribution
        f_radius = frequency_radius(fx, fy, ft, ft_0=ft_0, clean_division=True)
        env = 1./f_radius*np.exp(-.5*(np.log(f_radius/sf_0)**2)/((np.log((sf_0+B_sf)/sf_0))**2))
        return env
    else:
        return np.exp(-.5*(frequency_radius(fx, fy, ft, ft_0=ft_0) - sf_0)**2/B_sf**2)

def envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V):
    """
    Returns the speed envelope:
    selects the plane corresponding to the speed ``(V_X, V_Y)`` with some bandwidth ``B_V``.

    * (V_X, V_Y) = (0,1) is downward and  (V_X, V_Y) = (1, 0) is rightward in the movie.
    * A speed of V_X=1 corresponds to an average displacement of 1/N_X per frame.
    To achieve one spatial period in one temporal period, you should scale by
    V_scale = N_X/float(N_frame)
    If N_X=N_Y=N_frame and V=1, then it is one spatial period in one temporal
    period. It can be seen along the diagonal in the fx-ft face of the MC cube.

    A special case is used when ``B_V==0``, where the ``fx-ft`` plane is used as
    the speed plane: in that case it is desirable to set ``(V_X, V_Y)`` to ``(0, 0)``
    to avoid aliasing problems.

    Another special case is used when ``B_V==np.inf``, that is, every frame is
    different (white noise in the temporal domain).

    Run the 'test_speed' notebook to explore the speed parameters, see
    https://neuralensemble.github.io/MotionClouds/posts/testing-speed.html

    """
    if ft.shape[2]==1 or B_V==np.inf:
        env = np.ones_like(fx)
    elif B_V==0:
        env = np.zeros_like(fx)
        env[ft==0] = 1.
    else:
        f_radius = frequency_radius(fx, fy, ft, ft_0=ft_0, clean_division=True)
        env = np.exp(-.5*((ft+fx*V_X+fy*V_Y))**2/(B_V*f_radius)**2)
    return env

def envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta):
    """
    Returns the orientation envelope:
    We use a von-Mises distribution on the orientation:
    - mean orientation is ``theta`` (in radians),
    - ``B_theta`` is the bandwidth (in radians). It is equal to the standard deviation of the Gaussian
    envelope which approximate the distribution for low bandwidths. The Half-Width at Half Height is
    given by approximately np.sqrt(2*B_theta_**2*np.log(2)).

    Run 'testing-grating.py' notebook to see the effect of changing theta and B_theta, see
    https://neuralensemble.github.io/MotionClouds/posts/testing-grating.html

    """
    if B_theta is np.inf: # for large bandwidth, returns a strictly flat envelope
        enveloppe_orientation = np.ones_like(fx)
    elif theta==0 and B_theta==0:
        # N_X, N_Y, N_frame = fx.shape[0], fy.shape[1], ft.shape[2]
        enveloppe_orientation = np.zeros_like(fx)
        enveloppe_orientation[fy==0] = 1.
    else: # non pathological case
        angle = np.arctan2(fy, fx)
        enveloppe_orientation = np.exp(np.cos(2*(angle-theta))/4/B_theta**2)
    return enveloppe_orientation

def envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha, **kwargs):
    """
    Returns the Motion Cloud kernel, that is the product of:
        * a speed envelope
        * an orientation envelope
        * an orientation envelope

    """
    envelope = envelope_color(fx, fy, ft, alpha=alpha)
    envelope *= envelope_orientation(fx, fy, ft, theta=theta, B_theta=B_theta)
    envelope *= envelope_radial(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor)
    envelope *= envelope_speed(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V)
#     envelope *= retina(fx, fy, ft)
    return envelope

def random_cloud(envelope, seed=None, impulse=False, events=None, do_amp=True):
    """
    Returns a Motion Cloud movie as a 3D matrix from a given envelope.

    It first creates a random phase spectrum, multiplies with the envelope and
    then it computes the inverse FFT to obtain the spatiotemporal stimulus.

    Options are:
     * use a specific seed to specify the RNG's seed,
     * test the impulse response of the kernel by setting impulse to True
     * test the effect of randomizing amplitudes too by setting do_amp to True
shape

    # TODO : issue a warning if more than 10% of the energy of the envelope falls off the Fourier cube
    # TODO : use a safety sphere to ensure all orientations are evenly chosen

    """

    (N_X, N_Y, N_frame) = envelope.shape
    amps = 1.
    if impulse:
        fx, fy, ft = get_grids(N_X, N_Y, N_frame)
        phase = -2*np.pi*(N_X/2*fx + N_Y/2*fy + N_frame/2*ft)
        F_events = np.exp(1j * phase)
    elif events is None:
        np.random.seed(seed=seed)
        phase = 2 * np.pi * np.random.rand(N_X, N_Y, N_frame)
        F_events = np.exp(1j * phase)
        if do_amp:
            # see Galerne, B., Gousseau, Y. & Morel, J.-M. Random phase textures: Theory and synthesis. IEEE Transactions in Image Processing (2010). URL http://www.biomedsearch.com/nih/Random-Phase-Textures-Theory-Synthesis/20550995.html. (basically, they conclude "Even though the two processes ADSN and RPN have different Fourier modulus distributions (see Section 4), they produce visually similar results when applied to natural images as shown by Fig. 11.")
            F_events *= np.random.randn(N_X, N_Y, N_frame)
    else:
        F_events = np.fft.fftn( events[:, :, :] )
        F_events = np.fft.fftshift(F_events)

    Fz = F_events * envelope

    # de-centering the spectrum
    Fz = np.fft.ifftshift(Fz)
    Fz[0, 0, 0] = 0. # removing the DC component
    z = np.fft.ifftn(Fz).real
    return z

########################## Display Tools #######################################

def visualize(z_in, azimuth=25., elevation=30.,
    thresholds=[0.94, .89, .75, .5, .25, .1], opacities=[.9, .8, .7, .5, .2, .1],
#     thresholds=[0.94, .89, .75], opacities=[.99, .7, .2],
#     thresholds=[0.7, .5, .2], opacities=[.95, .5, .2],
    fourier_label = {'f_x':'f_x', 'f_y':'f_y', 'f_t':'f_t'},
    name=None, ext=ext, do_axis=True, do_grids=False, draw_projections=True,
    colorbar=False, f_N=2., f_tN=2., figsize=figsize, figpath=figpath, **kwargs):
    """

    Visualization of the Fourier spectrum by showing 3D contour plots at different thresholds

    parameters
    ----------
    z : envelope of the cloud

    """
    z = z_in.copy()
    N_X, N_Y, N_frame = z.shape
    fx, fy, ft = get_grids(N_X, N_Y, N_frame)

    # Normalize the amplitude.
    z /= z.max()

    from vispy import app, scene, use
    try:
        AffineTransform = scene.transforms.AffineTransform
    except:
        AffineTransform = scene.transforms.MatrixTransform

    use(app='pyglet', gl='pyopengl2')
    #from vispy.util.transforms import perspective, translate, rotate
    from vispy.color import Color
    transparent = Color(color='black', alpha=0.)
    import colorsys
    canvas = scene.SceneCanvas(size=figsize, bgcolor='white', dpi=450)
    view = canvas.central_widget.add_view()

    vol_data = np.rollaxis(np.rollaxis(z, 1), 2)
#         volume = scene.visuals.Volume(vol_data, parent=view.scene)#frame)
    center = scene.transforms.STTransform(translate=( -N_X/2, -N_Y/2, -N_frame/2))
#         volume.transform = center
#         volume.cmap = 'blues'

    if draw_projections:
        from vispy.color import Colormap
        cm = Colormap([(1.0, 1.0, 1.0, 1.0), 'k'])
        opts = {'parent':view.scene, 'cmap':cm, 'clim':(0., 1.)}

        energy_xy = np.rot90(np.max(z, axis=2)[:, ::-1], 3)[:, ::-1]
        fourier_xy = scene.visuals.Image(np.rot90(energy_xy), **opts)
        tr_xy = AffineTransform()
        tr_xy.rotate(90, (0, 0, 1))
        tr_xy.translate((N_X/2, -N_Y/2, -N_frame/2))
        fourier_xy.transform = tr_xy

        energy_xt = np.rot90(np.max(z, axis=1)[:, ::-1], 3)[::-1, ::-1]
        fourier_xt = scene.visuals.Image(energy_xt, **opts)
        tr_xt = AffineTransform()
        tr_xt.rotate(90, (1, 0, 0))
        tr_xt.translate((-N_X/2, N_Y/2, -N_frame/2))
        fourier_xt.transform = tr_xt

        energy_yt = np.max(z, axis=0)#[:, ::-1]
        fourier_yt = scene.visuals.Image(energy_yt, **opts)
        tr_yt = AffineTransform()
        tr_yt.rotate(90, (0, 1, 0))
        tr_yt.translate((-N_X/2, -N_Y/2, N_frame/2))
        fourier_yt.transform = tr_yt

    # Generate iso-surfaces at different energy levels
    surfaces = []
    for i_, (threshold, opacity) in enumerate(zip(thresholds, opacities)):
        surfaces.append(scene.visuals.Isosurface(z, level=threshold,
#                                         color=Color(np.array(colorsys.hsv_to_rgb(1.*i_/len(thresholds), 1., 1.)), alpha=opacity),
                                    color=Color(np.array(colorsys.hsv_to_rgb(.66, 1., 1.)), alpha=opacity),
                                    shading='smooth', parent=view.scene)
                                                )
        surfaces[-1].transform = center

    # Draw a sphere at the origin
    axis = scene.visuals.XYZAxis(parent=view.scene)
    for p in ([1, 1, 1, -1, 1, 1], [1, 1, -1, -1, 1, -1], [1, -1, 1, -1, -1, 1],[1, -1, -1, -1, -1, -1],
              [1, 1, 1, 1, -1, 1], [-1, 1, 1, -1, -1, 1], [1, 1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, -1],
              [1, 1, 1, 1, 1, -1], [-1, 1, 1, -1, 1, -1], [1, -1, 1, 1, -1, -1], [-1, -1, 1, -1, -1, -1]):
        line = scene.visuals.Line(pos=np.array([[p[0]*N_X/2, p[1]*N_Y/2, p[2]*N_frame/2], [p[3]*N_X/2, p[4]*N_Y/2, p[5]*N_frame/2]]), color='black', parent=view.scene)

    axisX = scene.visuals.Line(pos=np.array([[0, -N_Y/2, 0], [0, N_Y/2, 0]]), color='red', parent=view.scene)
    axisY = scene.visuals.Line(pos=np.array([[-N_X/2, 0, 0], [N_X/2, 0, 0]]), color='green', parent=view.scene)
    axisZ = scene.visuals.Line(pos=np.array([[0, 0, -N_frame/2], [0, 0, N_frame/2]]), color='blue', parent=view.scene)

    if do_axis:
        t = {}
        for text in ['f_x', 'f_y', 'f_t']:
            t[text] = scene.visuals.Text(fourier_label[text], parent=canvas.scene, face='Helvetica', color='black')
            t[text].font_size = 8
        t['f_x'].pos = canvas.size[0] // 3, canvas.size[1] - canvas.size[1] // 8
        t['f_y'].pos = canvas.size[0] - canvas.size[0] // 8, canvas.size[1] - canvas.size[1] // 6
        t['f_t'].pos = canvas.size[0] // 8, canvas.size[1] // 2

    cam = scene.TurntableCamera(elevation=elevation, azimuth=azimuth, up='z')
    cam.fov = 48
    cam.scale_factor = N_X * 1.8
    if do_axis: margin = 1.35
    else: margin = 1
    cam.set_range((-N_X/2*margin, N_X/2/margin), (-N_Y/2*margin, N_Y/2/margin), (-N_frame/2*margin, N_frame/2/margin))
    view.camera = cam

    render_im = canvas.render(size=figsize)
    app.quit()
    if not(name is None):
        import vispy.io as io
        io.write_png(name + ext, render_im)
    else:
        return render_im

def cube(im_in, azimuth=30., elevation=45., name=None,
         ext=ext, do_axis=True, show_label=True,
         cube_label = {'x':'x', 'y':'y', 't':'t'},
         colormap='gray', roll=-180., vmin=0., vmax=1.,
         figsize=figsize, **kwargs):

    """

    Visualization of the stimulus as a cube

    """
    im = im_in.copy()

    N_X, N_Y, N_frame = im.shape
    fx, fy, ft = get_grids(N_X, N_Y, N_frame)
    import numpy as np
    from vispy import app, scene, use
    try:
        AffineTransform = scene.transforms.AffineTransform
    except:
        AffineTransform = scene.transforms.MatrixTransform

    use(app='pyglet', gl='pyopengl2')
    from vispy.util.transforms import perspective, translate, rotate
    canvas = scene.SceneCanvas(size=figsize, bgcolor='white', dpi=450)
    view = canvas.central_widget.add_view()

#         frame = scene.visuals.Cube(size = (N_X/2, N_frame/2, N_Y/2), color=(0., 0., 0., 0.),
#                                         edge_color='k',
#                                         parent=view.scene)
    for p in ([1, 1, 1, -1, 1, 1], [1, 1, -1, -1, 1, -1], [1, -1, 1, -1, -1, 1],[1, -1, -1, -1, -1, -1],
              [1, 1, 1, 1, -1, 1], [-1, 1, 1, -1, -1, 1], [1, 1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, -1],
              [1, 1, 1, 1, 1, -1], [-1, 1, 1, -1, 1, -1], [1, -1, 1, 1, -1, -1], [-1, -1, 1, -1, -1, -1]):
#             line = scene.visuals.Line(pos=np.array([[p[0]*N_Y/2, p[1]*N_X/2, p[2]*N_frame/2], [p[3]*N_Y/2, p[4]*N_X/2, p[5]*N_frame/2]]), color='black', parent=view.scene)
        line = scene.visuals.Line(pos=np.array([[p[0]*N_X/2, p[1]*N_frame/2, p[2]*N_Y/2],
                                                [p[3]*N_X/2, p[4]*N_frame/2, p[5]*N_Y/2]]), color='black', parent=view.scene)

    opts = {'parent':view.scene, 'cmap':'grays', 'clim':(0., 1.)}
    image_xy = scene.visuals.Image(np.rot90(im[:, :, 0], 3), **opts)
    tr_xy = AffineTransform()
    tr_xy.rotate(90, (1, 0, 0))
    tr_xy.translate((-N_X/2, -N_frame/2, -N_Y/2))
    image_xy.transform = tr_xy

    image_xt = scene.visuals.Image(np.fliplr(im[:, -1, :]), **opts)
    tr_xt = AffineTransform()
    tr_xt.rotate(90, (0, 0, 1))
    tr_xt.translate((N_X/2, -N_frame/2, N_Y/2))
    image_xt.transform = tr_xt

    image_yt = scene.visuals.Image(np.rot90(im[-1, :, :], 1), **opts)
    tr_yt = AffineTransform()
    tr_yt.rotate(90, (0, 1, 0))
    tr_yt.translate((+N_X/2, -N_frame/2, N_Y/2))
    image_yt.transform = tr_yt

    if do_axis:
        t = {}
        for text in ['x', 'y', 't']:
            t[text] = scene.visuals.Text(cube_label[text], parent=canvas.scene, face='Helvetica', color='black')
            t[text].font_size = 8
        t['x'].pos = canvas.size[0] // 3, canvas.size[1] - canvas.size[1] // 8
        t['t'].pos = canvas.size[0] - canvas.size[0] // 5, canvas.size[1] - canvas.size[1] // 6
        t['y'].pos = canvas.size[0] // 12, canvas.size[1] // 2

    cam = scene.TurntableCamera(elevation=35, azimuth=30)
    cam.fov = 45
    cam.scale_factor = N_X * 1.7
    if do_axis: margin = 1.3
    else: margin = 1
    cam.set_range((-N_X/2, N_X/2), (-N_Y/2*margin, N_Y/2/margin), (-N_frame/2, N_frame/2))
    view.camera = cam
    if not(name is None):
        im = canvas.render(size=figsize)
        app.quit()
        import vispy.io as io
        io.write_png(name + ext, im)
    else:
        app.quit()
        return im

def check_if_anim_exist(filename, vext=vext, figpath=figpath, **kwargs):
    """
    Check if the movie already exists

    returns True if the movie does not exist, False if it does

    """
    return not(os.path.isfile(os.path.join(figpath, filename + vext)))

def anim_save(z, filename, display=True, vext=vext,
              T_movie=T_movie, verbose=False, **kwargs):
    """
    Saves a numpy 3D matrix (x-y-t) to a multimedia file.

    The input pixel values are supposed to lie in the [0, 1.] range.

    """

    import tempfile
#     from scipy.misc.pilutil import toimage
    import imageio
    if z.ndim == 4: # colored movie
        N_X, N_Y, three, N_frame = z.shape
    else: # grayscale
        N_X, N_Y, N_frame = z.shape
    fps = int(N_frame / T_movie)
    def make_frames(z):
        files = []
        tmpdir = tempfile.mkdtemp()

        if verbose:
            print('Saving sequence ' + filename + ' as a ' +  vext + ' format')
        for frame in range(N_frame):
            fname = 'frame%06d.png' % frame
            full_fname = os.path.join(tmpdir, fname)
            image = np.rot90(z[..., frame])
            imageio.imsave(full_fname, (image*255).astype(np.uint8), compression=0, quantize=256)
            files.append(fname)
        return tmpdir, files

    def test_ffmpeg():
        ret = os.system('ffmpeg -version')
        if not ret==0:
            raise Exception('Do you have ffmpeg installed in your PATH?')

    def remove_frames(tmpdir, files):
        """
        Remove frames from the temp folder

        """
        for fname in files: os.remove(os.path.join(tmpdir, fname))
        if not(tmpdir == None): os.rmdir(tmpdir)

    if verbose:
        verb_ = ''
    else:
        verb_ = ' 2>/dev/null'
    if vext == '.mpg':
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        test_ffmpeg()
        options = ' -f image2  -r ' + str(fps) + ' -y '
        os.system('ffmpeg -i ' + tmpdir + '/frame%06d.png ' + options + filename + vext + verb_)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.mp4': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        test_ffmpeg()
        options = ' -f mp4 -pix_fmt yuv420p -c:v libx264  -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%06d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.webm':
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        test_ffmpeg()
        options = ' -f webm  -pix_fmt yuv420p -vcodec libvpx -qmax 12 -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%06d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.mkv': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        test_ffmpeg()
        options = ' -y -f image2pipe -c:v png -i - -c:v libx264 -preset ultrafast -qp 0 -movflags +faststart -pix_fmt yuv420p  -g ' + str(fps) + '  -r ' + str(fps) + + ' -y '
        cmd = 'cat '  + tmpdir + '/*.png  | ffmpeg '  + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.gif': # http://www.uoregon.edu/~noeckel/MakeMovie.html
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        ret = os.system('convert -version')
        if not ret==0:
            raise Exception('Do you have convert installed in your PATH?')
        options = ' -set delay 8 -colorspace GRAY -colors 256 -dispose 1 -loop 0 '
        os.system('convert '  + tmpdir + '/frame*.png  ' + options + filename + vext + verb_)
        # 3) clean up
        remove_frames(tmpdir, files)

    elif vext == '.png':
        tmpdir, files = make_frames(z)
        import shutil
        shutil.copytree(tmpdir, filename)
        remove_frames(tmpdir, files)

    elif vext == '.zip':
        do_bmp = False # I was asked at some point to generate bmp files - it is highly unlikely to happen again...
        tmpdir, files = make_frames(z)
        import zipfile
        with zipfile.ZipFile(filename + vext, "w") as zf:
            if do_bmp:
                # convert to BMP for optical imaging
                files_bmp = []
                for fname in files:
                    fname_bmp = os.path.splitext(fname)[0] + '.bmp'
                    # generates 8-bit bmp (old format)
                    os.system('convert ' + fname + ' ppm:- | convert -size 256x256+0 -colors 256 -colorspace Gray - BMP2:' + fname_bmp)
                    files_bmp.append(fname_bmp)
                    zf.write(fname_bmp)
                remove_frames(tmpdir=None, files=files_bmp)
            else:
                for fname in files:
                    full_fname = os.path.join(tmpdir, fname)
                    zf.write(full_fname, arcname=fname)
        remove_frames(tmpdir, files)

    elif vext == '.mat':
        from scipy.io import savemat
        savemat(filename + vext, {'z':z})

    elif vext == '.npy':
        np.save(filename + vext, z)

    elif vext == '.h5':
        from tables import open_file, Float32Atom
        with open_file(filename + vext, 'w') as hf:
            o = hf.create_carray(hf.root, 'stimulus', Float32Atom(), z.shape)
            o = z
    else:
        print(' WARNING: extension ', vext , 'not existing! ')


def play(z, T=5.):
    """
    T: duration in second of a period

    TODO: currently failing on MacOsX - use numpyGL?

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
            print('FPS: %.2f (%d frames in %.2f seconds)' % (fps, frames, t-t0))
            frames, t0 = 0, t
         # computing the frame more closely to the actual time
        Z[...] = z[:, :, np.int(np.mod(t, T)/T * N_frame)].T.astype(np.float32)
        #Z[...] = z[:, :, frames % N_frame].T.astype(np.float32)
        image.update()
        fig.redraw()
    glumpy.show()

def figures_MC(fx, fy, ft, name='MC', V_X=V_X, V_Y=V_Y, do_figs=True, do_movie=True,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor, recompute=False,
                    theta=theta, B_theta=B_theta, alpha=alpha, vext=vext,
                    seed=None, impulse=False, do_amp=False, verbose=False, figpath=figpath, return_envelope = False, **kwargs):
    """
    Generates the figures corresponding to the Fourier spectra and the stimulus cubes and
    movies directly from the parameters.

    The figures names are automatically generated.

    """
    if not(os.path.isdir(figpath)): os.mkdir(figpath)
    z = envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                theta=theta, B_theta=B_theta, alpha=alpha)
    figures(z, name, vext=vext, do_figs=do_figs, do_movie=do_movie, recompute=recompute,
                    seed=seed, impulse=impulse, verbose=verbose, do_amp=do_amp, figpath=figpath, **kwargs)
    if return_envelope:
        return z

def figures(z=None, name='MC', vext=vext, do_movie=True, do_figs=True, recompute=False,
                    seed=None, impulse=False, events=None, verbose=False, masking=False, do_amp=False, figpath=figpath, **kwargs):
    """
    Given an envelope, generates the figures corresponding to the Fourier spectra
    and the stimulus cubes and movies.

    The figures names are automatically generated.

    """
    if not(os.path.isdir(figpath)): os.mkdir(figpath)

    if do_figs and not z is None:
        if recompute or check_if_anim_exist(name, vext=ext):
            try:
                visualize(z, name=os.path.join(figpath, name), **kwargs)           # Visualize the Fourier Spectrum
            except Exception as e:
                print('Failed to generate the visualisation:', e)

    if do_movie or do_figs:
            #if recompute:# or not(check_if_anim_exist(name, vext=vext) or check_if_anim_exist(name + '_cube', vext=ext)):
            movie = rectif(random_cloud(z, seed=seed, impulse=impulse, events=events, do_amp=do_amp), verbose=verbose)

    if do_figs:
        if recompute or check_if_anim_exist(name + '_cube', vext=ext):
            try:
                cube(movie, name=os.path.join(figpath, name + '_cube'), **kwargs)   # Visualize the Stimulus cube
            except Exception as e:
                print('Failed to generate the cube:', e)
    if do_movie:
        if recompute or check_if_anim_exist(name, vext=vext):
            try:
                anim_save(movie, os.path.join(figpath, name), display=False, vext=vext, **kwargs)
            except Exception as e:
                print('Failed to generate the movie:', e)

def in_show_video(name, vext=vext, loop=True, autoplay=True, controls=True, embed=False, figpath=figpath, **kwargs):
    """

    Columns represent isometric projections of a cube. The left column displays
    iso-surfaces of the spectral envelope by displaying enclosing volumes at 5
    different energy values with respect to the peak amplitude of the Fourier spectrum.
    The middle column shows an isometric view of the faces of the movie cube.
    The first frame of the movie lies on the x-y plane, the x-t plane lies on the
    top face and motion direction is seen as diagonal lines on this face (vertical
    motion is similarly see in the y-t face). The third column displays the actual
    movie as an animation.

    Given a name, displays the figures corresponding to the Fourier spectra, the
    stimulus cubes and movies within the notebook.

    """
    import os
    from IPython.core.display import display, Image, HTML
    from base64 import b64encode

    opts = ' '
    if loop: opts += 'loop="1" '
    if autoplay: opts += 'autoplay="1" '
    if controls: opts += 'controls '
    if embed:
        try:
            with open(os.path.join(figpath, name + ext), "rb") as image_file:
                im1 = b64encode(image_file.read()).decode("utf-8")
            with open(os.path.join(figpath, name + '_cube' + ext), "rb") as image_file:
                im2 = b64encode(image_file.read()).decode("utf-8")
            with open(os.path.join(figpath, name + vext), "rb") as video_file:
                im3 = b64encode(video_file.read()).decode("utf-8")

            s = """
            <center><table border=none width=100% height=100%>
            <tr>
            <td width=33%%><center><img src="data:image/png;base64,{0}" width=100%/></td>
            <td rowspan=2  colspan=2><center><video src="data:video/webm;base64,{1}"  {2}  type="video/{3}" width=100%/></td>
            </tr>
            <tr>
            <td><center><img src="data:image/png;base64,{4}" width=100%/></td>
            </tr>
            </table></center>""".format(im1, im3, opts, vext[1:], im2)
            # display(HTML(s))
        except:
            video = open(os.path.join(figpath, name + vext), "rb").read()
            video_encoded = b64encode(video).decode("utf-8")
            s = """
            <center><table border=none width=100% height=100%>
            <tr> <td width=100%><center><video {0} src="data:video/{1};base64,{2}" width=100%\>
            </td></tr></table></center>""".format(opts, vext[1:], video_encoded)
            # display(HTML(s))
    else:
        if os.path.isfile(os.path.join(figpath, name + ext)) and os.path.isfile(os.path.join(figpath, name + '_cube' + ext)):
            if os.path.isfile(os.path.join(figpath, name + vext)):
                s = """
                <center><table border=none width=100% height=100%>
                <tr>
                <td width=33%%><center><img src="{0}" width=100%/></td>
                <td rowspan=2  colspan=2><center><video src="{1}"  {2}  type="video/{3}" width=100%/></td>
                </tr>
                <tr>
                <td><center><img src="{4}" width=100%/></td>
                </tr>
                </table></center>""".format(os.path.join(figpath, name + ext),
                                      os.path.join(figpath, name + vext),
                                      opts, vext[1:],
                                      os.path.join(figpath, name + '_cube' + ext))
            else:
                s = """
                <center><table border=none width=100% height=100%>
                <tr>
                <td width=50%%><center><img src="{0}" width=100%/></td>
                <td><center><img src="{1}" width=100%/></td>
                </tr>
                </table></center>""".format(os.path.join(figpath, name + ext),
                                      os.path.join(figpath, name + '_cube' + ext))
        else:
            s = """
            <center><table border=none width=100% height=100%>
            <tr> <td width=100%><center><video {0} src="{2}" type="video/{1}"  width=100%\>
            </td></tr></table></center>""".format(opts, vext[1:], os.path.join(figpath, name + vext))
    html = HTML(s)
    html.reload()
    display(html)
