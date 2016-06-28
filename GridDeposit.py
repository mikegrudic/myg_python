from numba import jit, vectorize, float32, float64, autojit
import numpy as np

@vectorize([float32(float32), float64(float64)])
def Kernel2D(q):
    if q <= 0.5:
        return 1.8189136353359467 * (1 - 6*q**2 + 6*q**3)
    elif q <= 1.0:
        return 3.6378272706718935 * (1-q)**3
    else: return 0.0

@vectorize([float32(float32), float64(float64)])
def Kernel3D(q):
    if q <= 0.5:
        return 2.5464790894703255 * (1 - 6*q**2 + 6*q**3)
    elif q <= 1.0:
        return 2.5464790894703255 * 2 * (1-q)**3
    else: return 0.0

@jit#("f8[:,:](f8[:], f8[:], f8[:], f8, i8, f8)")
def GridDensity(mass, x, h, z, gridres, rmax):
    L = rmax*2
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i] + L/2
        hs = h[i]
        dz = (xs[2]-L/2)-z
        if np.abs(dz) > hs: continue

        mh3 = mass[i]/hs**3

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel3D((dz**2 + (xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                grid[gx,gy] +=  kernel * mh3
                
    return grid

@jit
def GridSurfaceDensity(mass, x, h, gridres, rmax):
    L = rmax*2
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i] + rmax
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel2D(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                grid[gx,gy] +=  kernel * mh2
                
    return grid

@jit
def GridProject(f, x, h, gridres, rmax):
    L = rmax*2
    Nf = len(f)
    grid = np.zeros((gridres,gridres, Nf))
    fh2 = np.zeros(len(f))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i] + rmax
        hs = h[i]
        for j in xrange(Nf):
            fh2[j] = f[j,i]/hs**2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel2D(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                for j in xrange(Nf):
                    grid[gx,gy, j] +=  kernel * fh2[j]
                    
#    for gx in xrange(gridres):
#        for gy in xrange(gridres):
#            for j in xrange(1,Nf):
#                grid[gx,gy,j] /= grid[gx,gy,0]

    return grid

@jit
def GridVariance(mass, F, x, h, gridres, rmax):
    L = rmax*2
    sigma = np.zeros((gridres,gridres))
    mean = np.zeros((gridres, gridres))
    Fsqr = np.zeros((gridres, gridres))
    result = np.zeros((gridres, gridres))
    
    dx = L/(gridres-1)
    N = len(x)
#First pass: get the mean
    for i in xrange(N):
        xs = x[i] + L/2
        hs = h[i]
        mh2 = mass[i]/hs**2
        fmh2 = F[i]*mh2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)

        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel2D(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                sigma[gx,gy] +=  kernel * mh2
                mean[gx,gy] +=  kernel * fmh2

#2nd pass: sum of squares
    for i in xrange(N):
        xs = x[i] + L/2
        hs = h[i]
        mh2 = mass[i]/hs**2
#        fmh2 = F[i]*mh2
#        f2mh2 = F[i]**2 * mh2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)

        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel2D(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                Fsqr[gx,gy] +=  kernel * mh2 * (F[i] - mean[gx,gy])**2

    for i in xrange(gridres):
        for j in xrange(gridres):
            if sigma[i,j] > 0:
                result[i,j] = Fsqr[i,j]/sigma[i,j]
            else:
                result[i,j] = 0.0
    return result

@jit("f8[:,:](f8[:], f8[:], f8[:], i8, f8)")
def GridSurfaceDensityPeriodic(mass, x, h, gridres, L):
    x = (x-L/2)%L
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i]
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = int((xs[0] - hs)/dx + 1)
        gxmax = int((xs[0] + hs)/dx)
        gymin = int((xs[1] - hs)/dx + 1)
        gymax = int((xs[1] + hs)/dx)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                ix = gx%gridres
                iy = gy%gridres
                delta_x = np.abs(xs[0] - ix*dx)
                delta_x = min(delta_x, L-delta_x)
                delta_y = np.abs(xs[1] - iy*dx)
                delta_y = min(delta_y, L-delta_y)
                kernel = Kernel2D((delta_x**2 + delta_y**2)**0.5 / hs)
                grid[ix,iy] +=  kernel * mh2
                 
    return grid
