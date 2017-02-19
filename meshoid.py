#MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Jesus H. Christ

import numpy as np
from scipy.spatial import cKDTree
from numba import jit, vectorize, float32, float64

class meshoid(object):
    def __init__(self, x, masses=None, des_ngb=None,boxsize=None, fixed_h = None): 
        if len(x.shape)==1:
            x = x[:,None]

        self.N, self.dim = x.shape
        if des_ngb==None:
            if fixed_h == None: 
                des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
            

        self.fixed_h = fixed_h
        self.des_ngb = des_ngb    
        print self.des_ngb    
        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        self.boxsize = boxsize
        
        if masses==None:
            masses = np.repeat(1./len(x),len(x))
        self.m = masses
        self.x = x
        
        self.TreeUpdate()
        
        self.dweights = None

    def ComputeWeights(self):
        dx = self.x[self.ngb] - self.x[:,None,:]

        if self.boxsize != None:
            Periodicize(dx.ravel(), self.boxsize)

        dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, dx, dx)
        
        dx_matrix = np.linalg.inv(dx_matrix)
        self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, dx, self.weights)

        dx2_matrix = np.einsum('ijk,ijl->ikl',dx,dx)
        dx2_matrix = np.einsum('ijk,ijl->ikl',dx2_matrix.flatten(), dx2_matrix.flatten())
        
        
    
    def TreeUpdate(self):
        if self.fixed_h == None:
            if self.dim == 1:
                sort_order = self.x[:,0].argsort()
                self.ngbdist, self.ngb = NearestNeighbors1D(self.x[:,0][sort_order], self.des_ngb)
                sort_order = invsort(sort_order)
                self.ngbdist, self.ngb = self.ngbdist[sort_order][0], self.ngb[sort_order][0]
            else:                
                self.tree = cKDTree(self.x, boxsize=self.boxsize)
                self.ngbdist, self.ngb = self.tree.query(self.x, self.des_ngb)
            self.h = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)
#        else:
#            self.h = np.repeat(self.fixed_h, len(self.x))
#            self.ngb = 
        q = np.einsum('i,ij->ij', 1/self.h, self.ngbdist)
        K = Kernel(q)
        self.weights = np.einsum('ij,i->ij',K, 1/np.sum(K,axis=1))
        self.density = self.des_ngb * self.m / (self.volnorm * self.h**self.dim)
        self.vol = self.m / self.density

    def D(self, f):
        df = DF(f, self.ngb)
        if self.dweights == None:
            self.ComputeWeights()
        return np.einsum('ijk,ij->ik',self.dweights,df)

    def Integrate(self, f):
        return np.einsum('i,i...->...', self.vol,f)

    def KernelVariance(self, f):
        return np.std(DF(f,self.ngb)*self.weights, axis=1)
    
    def KernelAverage(self, f):
        return np.einsum('ij,ij->i',self.weights, f[self.ngb])
        
@jit
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

@jit
def NearestNeighbors1D(x, des_ngb):
    N = len(x)
    neighbor_dists = np.empty((N,des_ngb))
    neighbors = np.empty((N,des_ngb),dtype=np.int64)
    for i in xrange(N):
        x0 = x[i]
        left = 0
#        if i == N-1:
#            right = 0
#        else:
#            right = 1
        right = 1
        total_ngb = 0
        while total_ngb < des_ngb:
            lpos = i - left
            rpos = i + right
            if lpos < 0:
                dl = 1e100
            else:
                dl = np.fabs(x0 - x[lpos])
            if rpos > N-1:
                dr = 1e100
            else:
                dr = np.fabs(x0 - x[rpos])

            if dl < dr:
                neighbors[i,total_ngb] = lpos
                neighbor_dists[i, total_ngb] = dl
                left += 1
            else:
                neighbors[i,total_ngb] = rpos
                neighbor_dists[i, total_ngb] = dr
                right += 1
            total_ngb += 1
    return neighbor_dists, neighbors

@jit
def invsort(index):
    out = np.empty_like(index)
    for i in xrange(len(index)):
        out[index[i]] = i
