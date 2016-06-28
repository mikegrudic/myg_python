#!/usr/bin/env python
"""
GHOST: Gizmo Hdf5 Output Slice and rayTrace

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

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from PlotSettings import *
from time import time
from PIL import Image, ImageDraw, ImageFont
from SnapData import SnapData                                   
import h5py
import numpy as np
from scipy import spatial, misc
from matplotlib.colors import LogNorm
import re
import SmoothingLength
from macros import macros
from docopt import docopt
from GridDeposit import *
import Fields
from Fields import *

arguments = docopt(__doc__)
filenames = arguments["<files>"]
rmax = float(arguments["--rmax"])
planes = arguments["--plane"].split(',')
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

#def ParsePlotString(pstring):
#    if "from" in pstring:
#        pstring, limits = pstring.split("from")
#        limits = [float(f) for f in limits.split('to')]
#    else:
#        expression = pstring
#        limits = None
#    if "//" in pstring:
#        numerator, denominator = pstring.split("//")
#    else:
#        numerator, denominator = pstring, None
#    return numerator, denominator, limits

def CoordTransform(coords, plane):
    if plane != 'z':
        x, y, z = coords.T
        return {"x": np.c_[y,z,x], "y": np.c_[x,z,y]}[plane]
    else:
        return coords            

for f in filenames:
    #Load the file and extract particle data
    data = SnapData(f, center=np.array([0,0,0]), periodic=False, verbose=verbose, n_ngb=32)
    
    #clip smoothing length to grid dx to avoid aliasing
    for d in data.particle_data:
        if "SmoothingLength" in d.keys():
            d["SmoothingLength"] = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)

#    plot_limits = {}
    #do our projections, slices and variances type by type
    for ptype, d in enumerate(data.particle_data):
        if not ptype in plots_todo.keys(): continue
        if len(plots_todo[ptype]) == 0: continue
        if not "Coordinates" in d.keys(): continue
        projfields = {}
        #make sure we have all the particle data we need and stick em all in one big array
        for field in plots_todo[ptype]:
            for r in requirements[field]:
                if not r in data.particle_data[ptype].keys():
                    getattr(Fields, "Compute"+r)(d)
                projfields[r] = d[r]
            
        pdata = np.array(projfields.values())
        pdata[1:] *= pdata[0]

        
        for plane in planes:
            griddata = GridProject(pdata, CoordTransform(d["Coordinates"],plane), d["SmoothingLength"], gridres, rmax)
            plt.imshow(np.log10(griddata[:,:,0]), cmap='inferno')
            plt.show()
        
        
        
