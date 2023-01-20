import numpy as np
from visualclustering import iVAT
from sklearn.metrics.pairwise import euclidean_distances
import random

def MM(x, cp):
    n, p = x.shape
    m = np.ones(cp)
    # d = np.sqrt(np.sum((x-x[0])**2, axis=1))
    d = np.linalg.norm(x-x[0], axis=1, ord=2) # ord=2 is for euclidean distance
    Rp = np.zeros((n, cp))
    Rp[:,0] = d
    for t in range(1, cp):
        d = np.minimum(d, Rp[:,t-1])
        m[t] = np.argmax(d)
        # Rp[:,t] = np.sqrt(np.sum(((x[int(m[t])] - x)**2), axis=1))
        Rp[:,t] = np.linalg.norm(x[int(m[t])] - x, axis=1)
    return m, Rp
 
def MMRS(x, cp, ns):
    n, p = x.shape
    m, rp = MM(x, cp)
    i = np.argmin(rp, axis=1)
    smp = []
    for t in range(cp):
        s = np.where(i==t)[0]
        nt = (np.ceil(ns*len(s)/n)).astype('int')
        # randomly sample nt points from s
        ind = random.sample(range(len(s)), nt)
        smp.append(s[ind])
        smp = [item for sublist in smp for item in sublist]
        smp = list(set(smp))
    return smp, rp, m
 
def clusivat(x,cp,ns):
    """ 
    x: data
    cp: number of clusters (over-estimated)
    ns: number of samples required from data
    """
    smp,rp,m = MMRS(x,cp,ns)
    rs = euclidean_distances(x[smp],x[smp])
    rv,C,I,ri,cut = iVAT(rs)
    return rv,C,I,ri,cut,smp