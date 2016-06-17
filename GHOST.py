#!/usr/bin/env python
"""
GHOST: Gadget Hdf5 Output Slice and rayTrace

  |\____
(:o ___/
  |/
  
Usage:
GHOST.py <files> ... [options]

Options:
    -h --help         Show this screen.
    --rmax=<kpc>      Maximum radius of plot window [default: 1.0]
    --plane=<x,y,z>   Slice/projection plane [default: z]
    --c=<cx,cy,cz>    Coordinates of plot window center [default: 0.0,0.0,0.0]
    --cmap=<name>     Name of colormap to use [default: viridis]
    --verbose         Verbose output
    --antialiasing    Using antialiasing when sampling the grid data for the actual plot. Costs some speed.
    --gridres=<N>     Resolution of slice/projection grid [default: 400]
    --neighbors=<N>   Number of neighbors used for smoothing length calculation [default: 32]
    --np=<N>          Number of processors to run on. [default: 1]
    --periodic        Must use for simulations in a periodic box.
    --imshow          Make an image without any axes instead of a matplotlib plot
    --notext          If imshow, do not add scale bar.
    --densweight      Produce image with intensity weighted by column density
    --blackholes      Plot black hole positions superimposed on plot
    --linscale        Use linear colormap
    --clusters
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PlotSettings import *
from PIL import Image, ImageDraw, ImageFont
from SnapData import SnapData                                   
import h5py
import numpy as np
from scipy import spatial, misc
from matplotlib.colors import LogNorm
import re
import SmoothingLength
from docopt import docopt
from GridDeposit import *

arguments = docopt(__doc__)
filenames = arguments["<files>"]
rmax = float(arguments["--rmax"])
plane = arguments["--plane"]
center = np.array([float(c) for c in re.split(',', arguments["--c"])])
verbose = arguments["--verbose"]
AA = arguments["--antialiasing"]
n_ngb = int(arguments["--neighbors"])
gridres = int(arguments["--gridres"])
nproc = int(arguments["--np"])
periodic = arguments["--periodic"]
colormap = arguments["--cmap"]
imshow = arguments["--imshow"]
notext = arguments["--notext"]
density_weighted = arguments["--densweight"]
blackholes = arguments["--blackholes"]
linscale = arguments["--linscale"]
plot_clusters = arguments["--clusters"]

font = ImageFont.truetype("LiberationSans-Regular.ttf", gridres/12)

if nproc > 1:
    from joblib import Parallel, delayed, cpu_count


data = SnapData(center=np.array([0,0,0]), periodic=False, verbose=False, n_ngb=32)
