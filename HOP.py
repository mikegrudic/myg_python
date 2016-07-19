#!/usr/bin/env python
import h5py
from numba import jit
import numpy as np
from sys import argv
from scipy import spatial, integrate
from matplotlib import pyplot as plt
type = 4
G = 4.3e4
brute_force_N = 1000
softening = 2.8e-4
min_cluster_size = 4
#type = 1
#G = 1

@jit
def ComputePotential(x, m):
    N = len(m)
    phi = np.zeros_like(m)
    for i in xrange(N):
        for j in xrange(N):
            if i==j: continue
            rij = ((x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2 + (x[i,2]-x[j,2])**2)**0.5
            phi[i] += m[j]/(rij**2 + softening**2)
    return -G*phi

@jit
def HsmlIter(neighbors, neighbor_dists, error_norm=1e-3):
    N, des_ngb = neighbor_dists.shape
    hsml = np.zeros(N)
    n_ngb = 0.0
    for i in xrange(N):
        upper = neighbor_dists[i,des_ngb-1]/0.5
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
            n_ngb *= 32./3
            if n_ngb > des_ngb:
                upper = h
            else:
                lower = h
            error = np.fabs(n_ngb-des_ngb)
        hsml[i] = h
    return hsml

@jit
def FindOwners(ngb, h):
    i = 0
    owners = -np.ones(len(h), dtype=np.int32)

    for i in xrange(len(h)):
        Owner(i, ngb, owners, h)
    return owners

@jit
def Owner(i, ngb, owners, h):
    if owners[i] > -1:
        return owners[i]
    elif ngb[i][h[ngb[i]].argmin()] == i:
        owners[i] = i
        return i
    else:
        owners[i] = Owner(ngb[i][h[ngb[i]].argmin()], ngb, owners, h)
        return owners[i]


F = h5py.File(argv[1])
if not "PartType4" in F.keys(): exit()
m = np.array(F["PartType%d"%type]["Masses"])
if len(m) < 1000: exit()
x = np.array(F["PartType%d"%type]["Coordinates"])
v = np.array(F["PartType%d"%type]["Velocities"])
t = np.array(F["PartType%d"%type]["StellarFormationTime"])

F.close()

def MakeBoundClusters(m, x, v, des_ngb = 50):
    def F(i):
        neighbors = owner_list[maxtree.query_ball_point(x[i], softening*10)]  

        if len(neighbors) == 1: return i
        elif np.prod(h[i] <= h[neighbors]):
            return i
        else:
            return F(neighbors[h[neighbors].argmin()])
    if len(m) < 100: return [],[],[],[],[]
    
    tree = spatial.cKDTree(x)
    ngbdist, ngb = tree.query(x,des_ngb)
    h = HsmlIter(ngb, ngbdist, error_norm=1e-8)

    owners = FindOwners(ngb[:,:8], h)
    owner_list = np.unique(owners)
    clusters = {}


    maxtree = spatial.cKDTree(x[owner_list])
    grouped_maxima = []

    refined_owners = [F(i) for i in owner_list]
    ro_map = dict(zip(owner_list, refined_owners))
    real_owners = [ro_map[i] for i in owners]

    clusters = {}
    for i in np.unique(real_owners):
        clusters[i] = []
    print "Assigning..."
    for i in xrange(len(x)):
        clusters[real_owners[i]].append(i)


    clusters = [c for c in clusters.values() if len(c) > min_cluster_size]

#print clusters[real_owners[0]]
    print "So far so good!"
    clusters = np.array(clusters)
    cluster_masses = np.array([m[c] for c in clusters])
    mtot = np.array([np.sum(M) for M in cluster_masses])
    cluster_coords = [x[c] for c in clusters]
    cluster_size = np.array([len(c) for c in clusters])
    clusters = clusters[mtot.argsort()]

#print cluster_size.min(), cluster_size.max()

    bound_clusters = []
    rejects = []
    r50s = []
    for c in clusters:
        xc, hc = x[c], h[c]
        center = np.percentile(xc, 50, axis=0)#np.average(xc,axis=0)#xc[hc.argmin()]
        R = spatial.distance.cdist(xc, [center,])[:,0]
        c = np.array(c)[R.argsort()]
        R = np.sort(R)
        xc, mc, hc, vc =  x[c], m[c], h[c], v[c]
#    vSqr = spatial.distance.cdist(vc, [vc[hc.argmin()]])[:,0]**2  #np.sum((v-vc)**2,axis=1)
#    vSqr = spatial.distance.cdist(vc, [np.average(vc,weights=mc,axis=0)])[:,0]**2
        vSqr = spatial.distance.cdist(vc, [np.percentile(vc,50,axis=0)])[:,0]**2 

        Mr = mc.cumsum()

        if len(c) < brute_force_N:
            phi = ComputePotential(xc, mc)
        else:
            phi = G*integrate.cumtrapz(Mr[::-1]/(R[::-1]**2 + softening**2), x=R[::-1], initial=0.0)[::-1] - G*mc.sum()/R[-1]
        rejects.append(c[0.5*vSqr >= -phi])
        bc = c[0.5*vSqr< -phi]

        if len(bc) > min_cluster_size:
            bound_clusters.append(bc)
            xc, hc = x[bc], h[bc]
            center = np.percentile(xc,50,axis=0)#[hc.argmin()]
            R = spatial.distance.cdist(xc, [center,])[:,0]
            r50 = np.percentile(R, 50)
            r50s.append(r50)

    rejects = np.concatenate(rejects)
#return bound_clusters, rejects

#for c in bound_clusters:
    r50s = np.array([r for r,c in zip(r50s,bound_clusters) if len(c) > min_cluster_size])
#clusters = np.array([c for c in clusters if len(c) > 16])
    mtot = np.array([np.sum(m[c]) for c in bound_clusters if len(c)>min_cluster_size])
    cluster_center = np.array([x[c][h[c].argmin()] for c in bound_clusters if len(c) > min_cluster_size])

    return bound_clusters, rejects, r50s, mtot, cluster_center

bound_clusters, rejects, r50s, mtot, cluster_center = MakeBoundClusters(m, x, v)
print len(rejects)
bound_clusters2, rejects2, r50s2, mtot2, cluster_center2 = MakeBoundClusters(m[rejects],x[rejects],v[rejects])

if len(bound_clusters2) > 0:
    rdict = dict(zip(np.arange(len(rejects)), rejects))
    bound_clusters = bound_clusters + [[rdict[p] for p in b] for b in bound_clusters2]
    rejects2 = [rdict[p] for p in rejects2]
    r50s = np.concatenate([r50s, r50s2])
    mtot = np.concatenate([mtot, mtot2])
    cluster_center = np.concatenate([cluster_center, cluster_center2])
#print len(rejects2), len(rejects)

data = np.c_[mtot, cluster_center, r50s, [len(c) for c in bound_clusters if len(c) > min_cluster_size]]
data = data[(-data[:,0]).argsort()]

n = argv[1].split("snapshot_")[1].split(".")[0]
np.savetxt(argv[1].split("snapshot")[0]+"clumps_%s.dat"%n, data)


Fout = h5py.File(argv[1].split("snapshot")[0] + "Clusters_%s.hdf5"%n, 'w')



Ncluster = np.array([len(c) for c in bound_clusters])
bound_clusters = np.array(bound_clusters)[Ncluster.argsort()[::-1]]

#reject_tree = spatial.cKDTree(x[rejects2])
#rmap = dict(zip(np.arange(len(rejects2)), rejects2))
#print "doing ball search..."
#for c in bound_clusters:
#    M = np.sum(m[c])
#    center = np.percentile(x[c],50,axis=0)
#    v0 = np.average(v[c],weights=m[c],axis=0)
#    ngb = reject_tree.query_ball_point(center, 0.01)
#    ngb = [rmap[n] for n in ngb]
#    if len(ngb) == 0: continue
#    R = np.sum((x[ngb]-center)**2, axis=1)**0.5
#    vSqr = np.sum((v[ngb]-v0)**2, axis=1)
#    bound = np.array(ngb)[vSqr < 2*G*M/R]
#    print bound
#    exit()
#print "Done!"
#bound_clusters = [c for n, c in sorted(zip(-Ncluster, bound_clusters))]

for i, c in enumerate(bound_clusters):
    cluster_id = "Cluster"+ ("%d"%i).zfill(int(np.log10(len(bound_clusters))+1))
    
    N = len(c)
    
    Fout.create_group(cluster_id)
    Fout[cluster_id].create_dataset("Masses", data = m[c])
    Fout[cluster_id].create_dataset("Coordinates", data = x[c])
    Fout[cluster_id].create_dataset("StellarFormationTime", data = t[c])
    Fout[cluster_id].create_dataset("Velocities", data=v[c])
    
    
Fout.close()
    
