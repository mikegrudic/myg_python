import numpy as np
from scipy.spatial import cKDTree
from numba import jit, vectorize, float32, float64

class ParticleSystem(object):
    def __init__(self, x, m=None, des_ngb=None,boxsize=None): 
#        if __name__ == "__main__": return

        if len(x.shape)==1:
            x = x[:,None]

        self.N, self.dim = x.shape
        if des_ngb is None:
            des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
            
        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        
        self.des_ngb = des_ngb
        self.boxsize = boxsize
        
        if m==None:
            m = np.repeat(1./len(x),len(x))
        self.m = m
        #print "Updating tree..."
        self.x = x
        self.TreeUpdate()
        self.density = self.des_ngb * self.m / (self.volnorm * self.h**self.dim)
    
    def D(self, f):
        df = DF(f, self.ngb)
        return np.einsum('ijk,ij->ik',self.dweights,df)
    
    def TreeUpdate(self):
        self.tree = cKDTree(self.x, boxsize=self.boxsize)
        self.ngbdist, self.ngb = self.tree.query(self.x, self.des_ngb)
        #print "Computing smoothing length..."

        self.h = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)

        #print "Computing grad weights..."
        self.q = np.einsum('i,ij->ij', 1/self.h, self.ngbdist)
    
        self.K = Kernel(self.q)
        #self.dK = DKernel(self.q)
        self.weights = np.einsum('ij,i->ij',self.K, 1/np.sum(self.K,axis=1))
        
        self.dx = self.x[self.ngb] - self.x[:,None,:]

        if self.boxsize != None:
            Periodicize(self.dx.ravel(), self.boxsize)

        dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, self.dx,self.dx)
        dx_matrix = np.linalg.inv(dx_matrix)
        self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, self.dx, self.weights)

#@jit
def HsmlIter(neighbor_dists,  dim=3, error_norm=1e-6):
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
        
@vectorize([float32(float32), float64(float64)])
def Kernel(q):
    if q <= 0.5:
        return 1 - 6*q**2 + 6*q**3
    elif q <= 1.0:
        return 2 * (1-q)**3
    else: return 0.0
        
@jit
def DF(f, ngb):
    df = np.empty(ngb.shape)
    for i in xrange(ngb.shape[0]):
        for j in xrange(ngb.shape[1]):
            df[i,j] = f[ngb[i,j]] - f[i]
    return df
    
@jit
def Periodicize(dx, boxsize):
    for i in xrange(dx.size):
        if np.abs(dx[i]) > boxsize/2:
            dx[i] = -np.sign(dx[i])*(boxsize - np.abs(dx[i]))