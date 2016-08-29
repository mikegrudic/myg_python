import numpy as np
from numba import jit, vectorize, float32, float64, autojit
from scipy.spatial import cKDTree

def SmoothingLength(coords, des_ngb=32, box_size=None, error_norm=1e-12):
    if box_size == None:
        tree = cKDTree(coords)
    else:
        tree = cKDTree((coords+box_size/2)%box_size, box_size=box_size)
    neighbor_dists, neighbors = tree.query(coords, des_ngb)

    hsml = HsmlIter(neighbor_dists,dim=coords.shape[1], error_norm = error_norm)
    return hsml

@jit
def HsmlIter(neighbor_dists,  dim=3, error_norm=1e-3):
    if dim==3:
        norm = 32./3
    elif dim==2:
        norm = 40./7
    else:
        norm = 8./3
    N, des_ngb = neighbor_dists.shape
    hsml = np.zeros(N)
    n_ngb = 0.0
    bound_coeff = (1./(1-(2*norm)**(-1./3)))
    for i in xrange(N):
        upper = neighbor_dists[i,des_ngb-1] * bound_coeff
        lower = neighbor_dists[i,1]
        error = 1e100
        count = 0
        while error > error_norm:
            h = (upper + lower)/2
            n_ngb=0.0
            dngb=0.0
            q = 0.0
            for j in xrange(des_ngb):
                q = neighbor_dists[i, j]/h
                if q <= 0.5:
                    n_ngb += (1 - 6*q**2 + 6*q**3)
                elif q <= 1.0:
                    n_ngb += 2*(1-q)**3
            n_ngb *= norm
            if n_ngb > des_ngb:
                upper = h
            else:
                lower = h
            error = np.fabs(n_ngb-des_ngb)
        hsml[i] = h
    return hsml
