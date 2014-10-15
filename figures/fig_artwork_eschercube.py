# coding: utf-8
"""

Cover Art for the Journal of Neurophysiology


"""

# Author: Paula Sanz Leon

# numeric
import numpy as np
# visualizers
from tvtk.api import tvtk
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi import mlab
import progressbar
import MotionClouds as mc

def source(dim, bwd):
    """
    Create motion cloud source
    """
    fx, fy, ft = dim
    z = mc.envelope_gabor(fx, fy, ft, B_sf=bwd[0], B_V=bwd[1], B_theta=bwd[2])
    data = mc.rectif(mc.random_cloud(z))
    return data


def image_data(dim, sp, orig, bwd):
    """
    Create Image Data for the surface from a data source.
    If rsrc == True, it generates random data.
    """
#    if rsrc:
#        data = random_source(dim)
#    else:
    data = source(dim, bwd)
    i = tvtk.ImageData(spacing=sp, origin=orig)
    i.point_data.scalars = data.ravel()
    i.point_data.scalars.name = 'scalars'
    i.dimensions = data.shape
    return i


def view(dataset):
    """
    Open up a mayavi scene and display the cubeset in it.
    """
    engine = mlab.get_engine()
    #fig = mlab.figure(bgcolor=(0, 0, 0), fgcolor=(1, 1, 1),
    #                  figure=dataset.class_name[3:])
    src = VTKDataSource(data=dataset)
    engine.add_source(src)
    # TODO : make some cubes more redish to show some "activity"
    mlab.pipeline.surface(src, colormap='gray')

def main(dim, sp=(1, 1, 1), orig=(0, 0, 0), B=(.01, .1, .4)):
    view(image_data(dim=dim, sp=sp, orig=orig, bwd=B))

infile = \
"""ATOM      1  N   ASN A   4      0   0  0   1.00 58.06           N
ATOM      2  CA  ASN A   4      0   0  1   1.00 57.60           C
ATOM      3  C   ASN A   4      0   0  2   1.00 56.99           C
ATOM      4  O   ASN A   4      0   0  3   1.00 56.93           O
ATOM      5  CB  ASN A   4      0   0  4   1.00 58.74           C
ATOM      6  CG  ASN A   4      0   1  0   1.00 59.79           C
ATOM      7  OD1 ASN A   4      0   1  1   1.00 61.28           O
ATOM      8  ND2 ASN A   4      0   1  2   1.00 58.25           N
ATOM      9  N   CYS A   5      0   1  3   1.00 55.63           N
ATOM     10  CA  CYS A   5      0   1  4   1.00 53.95           C
ATOM     11  C   CYS A   5      0   2  0   1.00 52.39           C
ATOM     12  O   CYS A   5      0   2  1   1.00 52.06           O
ATOM     13  CB  CYS A   5      0   2  2   1.00 55.11           C
ATOM     14  SG  CYS A   5      0   2  3   1.00 55.97           S
ATOM     15  N   GLU A   6      0   2  4   1.00 51.02           N
ATOM     16  CA  GLU A   6      0   3  0   1.00 49.76           C
ATOM     17  C   GLU A   6      0   3  1   1.00 48.26           C
ATOM     18  O   GLU A   6      0   3  2   1.00 47.48           O
ATOM     19  CB  GLU A   6      0   3  3   1.00 50.71           C
ATOM     20  CG  GLU A   6      0   3  4   1.00 52.02           C
ATOM     21  CD  GLU A   6      0   4  0   1.00 52.69           C
ATOM     22  OE1 GLU A   6      0   4  1   1.00 53.11           O
ATOM     23  OE2 GLU A   6      0   4  2   1.00 51.67           O
ATOM     24  N   ARG A   7      0   4  3   1.00 46.76           N
ATOM     25  CA  ARG A   7      0   4  4   1.00 44.82           C
ATOM     26  C   ARG A   7      1   0  0   1.00 42.48           C
ATOM     27  O   ARG A   7      1   0  1   1.00 41.67           O
ATOM     28  CB  ARG A   7      1   0  2   1.00 46.70           C
ATOM     29  CG  ARG A   7      1   0  3   1.00 49.35           C
ATOM     30  CD  ARG A   7      1   0  4   1.00 50.90           C
ATOM     31  NE  ARG A   7      1   1  0   1.00 52.89      N
ATOM     32  CZ  ARG A   7      1   1  1   1.00 54.22           C
ATOM     33  NH1 ARG A   7      1   1  2   1.00 54.43           N
ATOM     34  NH2 ARG A   7      1   1  3   1.00 54.93           N
ATOM     35  N   VAL A   8      1   1  4   1.00 40.07           N
ATOM     36  CA  VAL A   8      1   2  0   1.00 38.40           C
ATOM     37  C   VAL A   8      1   2  1   1.00 37.10           C
ATOM     38  O   VAL A   8      1   2  2   1.00 36.61           O
ATOM     39  CB  VAL A   8      1   2  3   1.00 38.49           C
ATOM     40  CG1 VAL A   8      1   2  4   1.00 37.92           C
ATOM     41  CG2 VAL A   8      1   3  0   1.00 37.46           C
ATOM     42  N   TRP A   9      1   3  1   1.00 35.63           N
ATOM     43  CA  TRP A   9      1   3  2   1.00 35.31           C
ATOM     44  C   TRP A   9      1   3  3   1.00 34.50           C
ATOM     45  O   TRP A   9      1   3  4   1.00 34.21           O
ATOM     46  CB  TRP A   9      1   4  0   1.00 35.21           C
ATOM     47  CG  TRP A   9      1   4  1   1.00 34.36           C
ATOM     48  CD1 TRP A   9      1   4  2   1.00 34.09           C
ATOM     49  CD2 TRP A   9      1   4  3   1.00 33.98           C
ATOM     50  NE1 TRP A   9      1   4  4   1.00 33.87           N
ATOM     51  CE2 TRP A   9      2   0  0   1.00 33.39           C
ATOM     52  CE3 TRP A   9      2   0  1   1.00 34.42           C
ATOM     53  CZ2 TRP A   9      2   0  2   1.00 34.38           C
ATOM     54  CZ3 TRP A   9      2   0  3   1.00 34.33           C
ATOM     55  CH2 TRP A   9      2   0  4   1.00 33.83           C
ATOM     56  N   LEU A  10      2   1  0   1.00 34.21           N
ATOM     57  CA  LEU A  10      2   1  1   1.00 33.50           C
ATOM     58  C   LEU A  10      2   1  2   1.00 33.86           C
ATOM     59  O   LEU A  10      2   1  3   1.00 33.20           O
ATOM     60  CB  LEU A  10      2   1  4   1.00 33.96           C
ATOM     61  CG  LEU A  10      2   2  0   1.00 33.96           C
ATOM     62  CD1 LEU A  10      2   2  1   1.00 35.37           C
ATOM     63  CD2 LEU A  10      2   2  2   1.00 33.69           C
ATOM     64  N   ASN A  11      2   2  3   1.00 32.80           N
ATOM     65  CA  ASN A  11      2   2  4   1.00 32.32           C
ATOM     66  C   ASN A  11      2   3  0   1.00 30.27           C
ATOM     67  O   ASN A  11      2   3  1   1.00 29.07           O
ATOM     68  CB  ASN A  11      2   3  2   1.00 33.12           C
ATOM     69  CG  ASN A  11      2   3  3   1.00 37.83           C
ATOM     70  OD1 ASN A  11      2   3  4   1.00 38.28           O
ATOM     71  ND2 ASN A  11      2   4  0   1.00 38.48           N
ATOM     72  N   VAL A  12      2   4  1   1.00 30.26           N
ATOM     73  CA  VAL A  12      2   4  2   1.00 29.84           C
ATOM     74  C   VAL A  12      2   4  3   1.00 28.89           C
ATOM     75  O   VAL A  12      2   4  4   1.00 27.95           O
ATOM     76  CB  VAL A  12      3   0  0   1.00 30.35           C
ATOM     77  CG1 VAL A  12      3   0  1   1.00 31.23           C
ATOM     78  CG2 VAL A  12      3   0  2   1.00 32.15           C
ATOM     79  N   THR A  13      3   0  3   1.00 27.12           N
ATOM     80  CA  THR A  13      3   0  4   1.00 26.31           C
ATOM     81  C   THR A  13      3   1  0   1.00 25.85           C
ATOM     82  O   THR A  13      3   1  1   1.00 24.34           O
ATOM     83  CB  THR A  13      3   1  2   1.00 25.95           C
ATOM     84  OG1 THR A  13      3   1  3   1.00 28.37           O
ATOM     85  CG2 THR A  13      3   1  4   1.00 25.87           C
ATOM     86  N   PRO A  14      3   2  0   1.00 26.14           N
ATOM     87  CA  PRO A  14      3   2  1   1.00 25.69           C
ATOM     88  C   PRO A  14      3   2  2   1.00 25.90           C
ATOM     89  O   PRO A  14      3   2  3   1.00 25.31           O
ATOM     90  CB  PRO A  14      3   2  4   1.00 27.24           C
ATOM     91  CG  PRO A  14      3   3  0   1.00 25.82           C
ATOM     92  CD  PRO A  14      3   3  1   1.00 26.62           C
ATOM     93  N   ALA A  15      3   3  2   1.00 23.79           N
ATOM     94  CA  ALA A  15      3   3  3   1.00 23.86           C
ATOM     95  C   ALA A  15      3   3  4   1.00 23.33           C
ATOM     96  O   ALA A  15      3   4  0   1.00 22.73           O
ATOM     97  CB  ALA A  15      3   4  1   1.00 23.33           C
ATOM     98  N   THR A  16      3   4  2   1.00 23.23           N
ATOM     99  CA  THR A  16      3   4  3   1.00 23.30           C
ATOM    100  C   THR A  16      3   4  4   1.00 23.54           C
ATOM    101  O   THR A  16      4   0  0   1.00 24.07           O
ATOM    102  CB  THR A  16      4   0  1   1.00 23.08           C
ATOM    103  OG1 THR A  16      4   0  2   1.00 23.79           O
ATOM    104  CG2 THR A  16      4   0  3   1.00 21.42           C
ATOM    105  N   LEU A  17      4   0  4   1.00 22.68           N
ATOM    106  CA  LEU A  17      4   1  0   1.00 22.52           C
ATOM    107  C   LEU A  17      4   1  1   1.00 24.49           C
ATOM    108  O   LEU A  17      4   1  2   1.00 24.40           O
ATOM    109  CB  LEU A  17      4   1  3   1.00 22.34           C
ATOM    110  CG  LEU A  17      4   1  4   1.00 21.43           C
ATOM    111  CD1 LEU A  17      4   2  0   1.00 22.89           C
ATOM    112  CD2 LEU A  17      4   2  1   1.00 20.09           C
ATOM    113  N   ARG A  18      4   2  2   1.00 25.87           N
ATOM    114  CA  ARG A  18      4   2  3   1.00 28.13           C
ATOM    115  C   ARG A  18      4   2  4   1.00 28.16           C
ATOM    116  O   ARG A  18      4   3  0   1.00 27.66           O
ATOM    117  CB  ARG A  18      4   3  1   1.00 29.75           C
ATOM    118  CG  ARG A  18      4   3  2   1.00 34.31           C
ATOM    119  CD  ARG A  18      4   3  3   1.00 37.23           C
ATOM    120  NE  ARG A  18      4   3  4   1.00 41.66           N
ATOM    121  CZ  ARG A  18      4   4  0   1.00 46.11           C
ATOM    122  NH1 ARG A  18      4   4  1   1.00 46.12           N
ATOM    123  NH2 ARG A  18      4   4  2   1.00 48.66           N
ATOM    124  N   SER A  19      4   4  3   1.00 26.84           N
ATOM    125  CA  SER A  19      4   4  4   1.00 28.90           C
CONECT  1  2  3  4  5
CONECT  6  7  8  9  10
CONECT  11  12  13  14  15
CONECT  16  17  18  19  20
CONECT  21  22  23  24  25
CONECT  26  27  28  29  30
CONECT  31  32  33  34  35
CONECT  36  37  38  39  40
CONECT  41  42  43  44  45
CONECT  46  47  48  49  50
CONECT  51  52  53  54  55
CONECT  56  57  58  59  60
CONECT  61  62  63  64  65
CONECT  66  67  68  69  70
CONECT  71  72  73  74  75
CONECT  76  77  78  79  80
CONECT  81  82  83  84  85
CONECT  86  87  88  89  90
CONECT  91  92  93  94  95
CONECT  96  97  98  99 100
CONECT  101  102  103  104  105
CONECT  106  107  108  109  110
CONECT  111  112  113  114  115
CONECT  116  117  118  119  120
CONECT  121  122  123  124  125
CONECT  1  6 11  16  21
CONECT  26  31  36  41  46
CONECT  51  56  61  66  71
CONECT  76  81  86  91  96
CONECT  101  106  111  116  121
CONECT  2  7  12  17  22
CONECT  27  32  37  42  47
CONECT  52  57  62  67  72
CONECT  77  82  87  92  97
CONECT  102  107  112  117  122
CONECT  3  8   13  18  23
CONECT  28  33  38  43  48
CONECT  53  58  63  68  73
CONECT  78  83  88  93  98
CONECT  103  108  113  118  123
CONECT  4  9  14  19  24
CONECT  29  34  39  44  49
CONECT  54  59  64  69  74
CONECT  79  84  89  94  99
CONECT  104  109  114  119 124
CONECT  5  10  15  20  25
CONECT  30  35  40  45  50
CONECT  55  60  65  70  75
CONECT  80  85  90  95  100
CONECT  105  110  115  120  125
CONECT  1  26  51  76  101
CONECT  2  27  52  77  102
CONECT  3  28  53  78  103
CONECT  4  29  54  79  104
CONECT  5  30  55  80  105
CONECT  6  31  56  81  106
CONECT  7  32  57  82  107
CONECT  8  33  58  83  108
CONECT  9  34  59  84  109
CONECT  10 35  60  85  110
CONECT  11 36  61  86  111
CONECT  12 37  62  87  112
CONECT  13 38  63  88  113
CONECT  14 39  64  89  114
CONECT  15 40  65  90  115
CONECT  16 41  66  91  116
CONECT  17 42  67  92  117
CONECT  18 43  68  93  118
CONECT  19 44  69  94  119
CONECT  20 45  70  95  120
CONECT  21 46  71  96  121
CONECT  22 47  72  97  122
CONECT  23 48  73  98  123
CONECT  24 49  74  99  124
CONECT  25 50  75  100 125
"""
def get_nodes_and_edges():
    # protein_code = 'mc0612'
    # import gzip
    # infile = gzip.GzipFile('%s.ent.gz' % protein_code, 'rb')

    # A graph represented by a dictionary associating nodes with keys
    # (numbers), and edges (pairs of node keys).
    nodes = dict()
    edges = list()
    atoms = set()
    # Build the graph from the PDB information
    for line in infile.splitlines():
        line = line.split()
        if line[0] in ('ATOM', 'HETATM'):
            nodes[line[1]] = (line[2], line[6], line[7], line[8])
            atoms.add(line[2])
        elif line[0] == 'CONECT':
            for start, stop in zip(line[1:-1], line[2:]):
                edges.append((start, stop))

    atoms = list(atoms)
    atoms.sort()
    atoms = dict(zip(atoms, range(len(atoms))))

    # Turn the graph into 3D positions, and a connection list.
    labels = dict()

    x       = list()
    y       = list()
    z       = list()
    scalars = list()

    for index, label in enumerate(nodes):
        labels[label] = index
        this_scalar, this_x, this_y, this_z= nodes[label]
        scalars.append(atoms[this_scalar])
        x.append(float(this_x))
        y.append(float(this_y))
        z.append(float(this_z))

    connections = list()

    for start, stop in edges:
        #import pdb; pdb.set_trace()
        connections.append((labels[start], labels[stop]))

    x       = np.array(x)
    y       = np.array(y)
    z       = np.array(z)
    scalars = np.array(scalars)
    return x, y, z, connections, scalars



if __name__ == '__main__':

    #legend
    """

   Motion Clouds are a set of stimuli designed to explore in a systematic
    way the functional response of a sensory system to a natural-like motion
    stimulus. These are optimized for translating, full-field motion and are by
    construction textures synthesized from randomly placed similar motion
    patches with characteristics spatial parameters. The object of such an
    endeavor is to systematically test a system to varying the parameters by
    testing the response to a series of such textures. We show here for a fixed
    set of central parameters (mean speed, direction, orientation and spatial
    frequency) a cube constituted by a family of such Motion Clouds when varying
    the bandwidth parameters for speed (left panel), frequency (right panel) and
     orientation (top panel). Each elementary cube in the larger cube denoting
    the parameter space represents a Motion Cloud and is shown as a cube to
    represent the corresponding movie, with time flowing from lower left to
    upper right in the right and top facets. Overlaid hue gives a measure of a
    typical response for a sensory response (here a motion energy model) which
    gives a complete characterization of the sensory system at hand.

    Les \emph{Motion Clouds} constituent un ensemble de stimuli visant à explorer de manière systématique la réponse fonctionnelle d'un système sensoriel à un stimulus en mouvement de type naturel. Ceux-ci sont optimisées pour décrire un mouvement de translation pure en plein champ et sont par construction des textures synthétisées à partir de ``patchs" élémentaires de mouvement semblable placés au hasard dans l'espace. L'objet d'une telle entreprise est de tester systématiquement un système en variant les paramètres de telles textures sur les dimensions perceptives principales (vitesse moyenne,  direction, l'orientation spatiale et  fréquence). Nous montrons ici un \emph{espèce de stimuli} comme une grille tri-dimensionnelle dont les n\oe dus correspondent à des stimuli et les axes des paramètres du mouvement: bande passante pour la vitesse (panneau de gauche), fréquence (panneau de droite) et orientation (partie supérieure). Chaque n\oe ud contient un cube élémentaire qui représente le film correspondant au stimulus, avec le temps qui s'écoule du coin inférieur gauche au coin en haut à droite dans les facettes à droite et en haut. Nous avons superposé en couleur une teinte qui représente une mesure de la réponse sensorielle (ici un modèle d'énergie du mouvement) dans cet espace de stimuli. Ce genre de caractérisation permet une étude systématique du système (ici oculomoteur) qui est étudié.
    """

    import itertools
#     size = 2**4
#     size = 2**5
    size = 2**6
#     size = 2**7

    space = 1.5 * size # space between 2 cubes
    N = 5
    idx = np.arange(N)
    pos = idx * space
    Bf = np.logspace(-2., 0.1, N)
    Bv = [0.01, 0.1, 0.5, 1.0, 10.0]
    sigma_div = [2, 3, 5, 8, 13] # I love the golden number :-)
    Bo = np.pi/np.array(sigma_div)

    fx, fy, ft = mc.get_grids(size, size, size)

    # x-axis = B_f
    # y-axis = B_V
    # z_axis = B_o
    downscale = 1
    # downscale =2
    # downscale = 1./2

    mlab.figure(1, bgcolor=(.6, .6, .6), fgcolor=(0, 0, 0), size=(1600/downscale,1200/downscale))
    mlab.clf()

    do_grid, do_splatter, do_MCs = False, True, False
    do_grid, do_splatter, do_MCs = True, True, True
    ################################################################################
    ##  Generate the cubical graph structure to connect each individual cube      ##
    ################################################################################
    if do_grid:
        x, y, z, connections, scalars = get_nodes_and_edges()

        x = x * space + size / 2.
        y = y * space + size / 2.
        z = z * space + size / 2.
        #scalars = np.random.uniform(low=0.3, high=0.55, size=x.shape)
        scalars = 0.8*np.ones(x.shape)

        pts = mlab.points3d(x, y, z, scalars, colormap='Blues', scale_factor=5.0, resolution=10)
        pts.mlab_source.dataset.lines = np.array(connections)

        # Use a tube filter to plot tubes on the link, varying the radius with the
        # scalar value
        tube = mlab.pipeline.tube(pts, tube_sides=15, tube_radius=1.5)
        #tube.filter.radius_factor = 1.
        #tube.filter.vary_radius = 'vary_radius_by_scalar'
        mlab.pipeline.surface(tube, color=(0.4, 0.4, 0.4) )#colormap='gray')

    ################################################################################
    ##                      Gaussian Splatter                                     ##
    ################################################################################
    if do_splatter:

    # Visualize the local atomic density
    #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))
#    gs = mlab.pipeline.gaussian_splatter(pts)
#    gs.filter.radius = 0.95
#    iso=mlab.pipeline.iso_surface(gs, colormap = 'RdBu', opacity=0.03)
#    iso.contour.number_of_contours = 50
#    gsvol = mlab.pipeline.volume(gs)
        x_, y_, z_ = np.mgrid[0:(N*space), 0:(N*space), 0:(N*space)]
#        response = np.exp(-(  ((x_-2*space) + (y_-3*space))**2 +((y_-3*space))**2 +((z_-2*space)/2.)**2)/2/(3.*space)**2)#-1/2.*np.exp(-(x_ + y_  +z_/2.- 5.)**2/2/(2.*space)**2)
        response = np.exp(-((x_-4*space)**2 +(y_- 1*space)**2 +((z_-3.4*space)/2.)**2)/2/(1.3*space)**2)#-1/2.*np.exp(-(x_ + y_  +z_/2.- 5.)**2/2/(2.*space)**2)
        #sf = mlab.pipeline.scalar_field(x_, y_, z_, response)
        #vol = mlab.pipeline.volume(sf, vmin=0, vmax = 4.)#, color='red')#, colormap = 'RdBu') #response.min()+0.65*(response.max()-response.min()),  vmax=min+0.9*(max-min))
        mlab.contour3d(response, opacity=0.5)
#        # Changing the ctf:
#        from tvtk.util.ctf import ColorTransferFunction
#        ctf = ColorTransferFunction()
#        ctf.add_rgb_point(0., 0., 0., 1.) # r, g, and b are float between 0 and 1
#        ctf.add_rgb_point(1., 1., 0., 0.) # r, g, and b are float between 0 and 1
##        ctf.add_hsv_point(value, h, s, v)
#        # ...
#        vol._volume_property.set_color(ctf)
#        vol._ctf = ctf
#        vol.update_ctf = True
#
#        # Changing the otf:
#        from tvtk.util.ctf import PiecewiseFunction
#        otf = PiecewiseFunction()
#        otf.add_point(0., 0.)
#        otf.add_point(1., 1.)
#        vol._otf = otf
#        vol._volume_property.set_scalar_opacity(otf)

    ################################################################################
    ##                      Generate the Motion Clouds cubes                      ##
    ################################################################################
    if do_MCs:
        print size
        for i, j, k in list(itertools.product(idx, idx, idx)):
            main(dim=(fx, fy, ft), orig=(pos[i], pos[j], pos[k]), B=(Bf[i], Bv[k], Bo[j]))

#    mlab.show(stop=True)
    elevation = 90. - 18.
    view = mlab.view(azimuth=290., elevation=elevation, distance='auto', focalpoint='auto')
    distance = view[2]
    mlab.savefig('MCartwork.png')
    if True:#False:
        N_frame = 256 #128
        widgets = ["calculating", " ", progressbar.Percentage(), ' ',
                   progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=N_frame).start()
        for i_az, azimuth in enumerate(np.linspace(0, 360, N_frame, endpoint=False)):
            mlab.view(*view)
            mlab.view(azimuth=azimuth)
            mlab.view(distance=distance*(.7 + 0.3*np.cos(azimuth*2*np.pi/360.)))
            mlab.savefig('_MCartwork%03d.png' % i_az, size=(1600/downscale, 1200/downscale)) #magnification=0.5)
            pbar.update(i_az)
        pbar.finish()
        import os
        os.system('ffmpeg -y -i _MCartwork%03d.png  MCartwork.mpg')
        os.system('ffmpeg -y -i _MCartwork%03d.png  MCartwork.mp4')
        #os.system('brew install gifsicle')
        #os.system('ffmpeg -y  -set delay 8 -pix_fmt rgb24 -r 12 -i _MCartwork%03d.png  MCartwork.gif')
        os.system('convert -delay 8 -loop 0 -colors 256  -scale 25% -layers Optimize  _MCartwork*.png  MCartwork.gif')
        os.system('rm _MCartwork*') #
