########################## Display Tools #######################################
# /usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
from .param import *
from .MotionClouds import *

import pyprind as progressbar

if not(os.path.isdir(figpath)): os.mkdir(figpath)

os.environ['ETS_TOOLKIT'] = 'qt4' # Works in Mac
# os.environ['ETS_TOOLKIT'] = 'wx' # Works in Debian

def import_mayavi():
    """
    Mayavi is difficult to compile on some architectures (think Win / Mac Os), so we
    allowed the possibility of an ``ImportError`` or even to avoid importing it
    at all by setting the ``MAAYAVI`` string to ``Avoid``.

    """
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
# Trick from http://github.enthought.com/mayavi/mayavi/tips.html :
# to use offscreen rendering, try ``xvfb :1 -screen 0 1280x1024x2`` in one terminal,
# then ``export DISPLAY=:1`` before you run your script

def visualize(z_in, azimuth=290., elevation=45.,
    thresholds=[0.94, .89, .75, .5, .25, .1], opacities=[.9, .8, .7, .5, .2, .1],
    name=None, ext=ext, do_axis=True, do_grids=False, draw_projections=True,
    colorbar=False, f_N=2., f_tN=2., figsize=figsize):
    """

    Visualization of the Fourier spectrum by showing 3D contour plots at different thresholds

    parameters
    ----------
    z : envelope of the cloud

    """
    import_mayavi()
    z = z_in.copy()
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

def cube(im_in, azimuth=-45., elevation=130., roll=-180., name=None,
         ext=ext, do_axis=True, show_label=True, colormap='gray',
         vmin=0., vmax=1., figsize=figsize):

    """

    Visualization of the stimulus as a cube

    """
    import_mayavi()
    im = im_in.copy()

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

def anim_save(z, filename, display=True, vext=vext,
              centered=False, T_movie=T_movie, verbose=True):
    """
    Saves a numpy 3D matrix (x-y-t) to a multimedia file.

    The input pixel values are supposed to lie in the [0, 1.] range.

    """
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
        options = ' -f mp4 -pix_fmt yuv420p -c:v libx264  -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    if vext == '.webm':
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        options = ' -f webm  -pix_fmt yuv420p -vcodec libvpx -qmax 12 -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
        cmd = 'ffmpeg -i '  + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_
        os.system(cmd)
        # 3) clean up
        remove_frames(tmpdir, files)

    if vext == '.mkv': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
        # 1) create temporary frames
        tmpdir, files = make_frames(z)
        # 2) convert frames to movie
        options = ' -y -f image2pipe -c:v png -i - -c:v libx264 -preset ultrafast -qp 0 -movflags +faststart -pix_fmt yuv420p  -g ' + str(fps) + '  -r ' + str(fps) + + ' -y '
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
            print 'FPS: %.2f (%d frames in %.2f seconds)' % (fps, frames, t-t0)
            frames, t0 = 0, t
         # computing the frame more closely to the actual time
        Z[...] = z[:, :, np.int(np.mod(t, T)/T * N_frame)].T.astype(np.float32)
        #Z[...] = z[:, :, frames % N_frame].T.astype(np.float32)
        image.update()
        fig.redraw()
    glumpy.show()

def rectif(z_in, contrast=.9, method='Michelson', verbose=False):
    """
    Transforms an image (can be 1,2 or 3D) with normal histogram into
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

def figures_MC(fx, fy, ft, name, V_X=V_X, V_Y=V_Y, do_figs=True, do_movie=True,
                    B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                    theta=theta, B_theta=B_theta, alpha=alpha, vext=vext, recompute=False,
                    seed=None, impulse=False, do_amp=False, verbose=False):
    """
    Generates the figures corresponding to the Fourier spectra and the stimulus cubes and
    movies directly from the parameters.

    The figures names are automatically generated.

    """
    z = envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                B_V=B_V, sf_0=sf_0, B_sf=B_sf, loggabor=loggabor,
                theta=theta, B_theta=B_theta, alpha=alpha)
    figures(z, name, vext=vext, do_figs=do_figs, do_movie=do_movie,
                    seed=seed, impulse=impulse, verbose=verbose, do_amp=do_amp)

def figures(z=None, name='MC', vext=vext, do_movie=True, do_figs=True, recompute=False,
                    seed=None, impulse=False, verbose=False, masking=False, do_amp=False):
    """
    Given an envelope, generates the figures corresponding to the Fourier spectra
    and the stimulus cubes and movies.

    The figures names are automatically generated.

    """

    if (MAYAVI == 'Import') and do_figs: import_mayavi()

    if (MAYAVI[:2]=='Ok') and do_figs:
        if recompute or check_if_anim_exist(name, vext=ext):
            visualize(z, name=os.path.join(figpath, name))           # Visualize the Fourier Spectrum

    if do_movie or ((MAYAVI[:2]=='Ok') and do_figs):
            #if recompute:# or not(check_if_anim_exist(name, vext=vext) or check_if_anim_exist(name + '_cube', vext=ext)):
            movie = rectif(random_cloud(z, seed=seed, impulse=impulse, do_amp=do_amp), verbose=verbose)

    if (MAYAVI[:2]=='Ok') and do_figs:
        if recompute or check_if_anim_exist(name + '_cube', vext=ext):
            cube(movie, name=os.path.join(figpath, name + '_cube'))   # Visualize the Stimulus cube

    if (do_movie):
        if recompute or check_if_anim_exist(name, vext=vext):
            anim_save(movie, os.path.join(figpath, name), display=False, vext=vext)

def in_show_video(name, loop=True, autoplay=True, controls=True, embed=True):
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
            <td rowspan=2  colspan=2><center><video src="%s"  %s  type="video/%s" width=100%%/></td>
            </tr>
            <tr>
            <td><center><img src="%s" width=100%%/></td>
            </tr>
            </table></center>"""%(im1, im3, opts, vext[1:], im2)
            display(HTML(s))
        except:
            video = open(os.path.join(figpath, name + vext), "rb").read()
            video_encoded = b64encode(video)
            s = """
            <center><table border=none width=100%% height=100%%>
            <tr> <td width=100%%><center><video {0} src="data:video/{1};base64,{2}" width=100%%\>
            </td></tr></table></center>'.format(opts, vext[1:], video_encoded)
            """
            display(HTML(s))
    else:
        if os.path.isfile(os.path.join(figpath, name + ext)) and os.path.isfile(os.path.join(figpath, name + '_cube' + ext)):
            s = """
            <center><table border=none width=100%% height=100%%>
            <tr>
            <td width=33%%><center><img src="%s" width=100%%/></td>
            <td rowspan=2  colspan=2><center><video src="%s"  %s  type="video/%s" width=100%%/></td>
            </tr>
            <tr>
            <td><center><img src="%s" width=100%%/></td>
            </tr>
            </table></center>"""%(os.path.join(figpath, name + ext),
                                  os.path.join(figpath, name + vext),
                                  opts, vext[1:],
                                  os.path.join(figpath, name + '_cube' + ext))
            display(HTML(s))
        else:
            s = """
            <center><table border=none width=100%% height=100%%>
            <tr> <td width=100%%><center><video {0} src="{2}" type="video/{1}"  width=100%%\>
            </td></tr></table></center>""".format(opts, vext[1:], os.path.join(figpath, name + vext))
            display(HTML(s))
