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

def optimized_MM(x, cp, distance_type='euclidean'):
    """
    The function implements the MaxMin (MM) algorithm for determining the "representative" or "prototypical" points in a dataset.
    
    Parameters:
    x (numpy.ndarray): 2D array representing the dataset.
    cp (int): Integer representing the number of representative points to be selected.
    distance_type (str): Type of distance to be used. 'euclidean' or 'cosine'. Default is 'euclidean'.
    
    Returns:
    m (numpy.ndarray): 1D array of indices of the selected representative points.
    Rp (numpy.ndarray): 2D array where each row corresponds to a point in the dataset and each column corresponds to a representative point, representing the distance between the point and the representative point.
    
    """
    n, p = x.shape
    m = np.ones(cp, dtype=np.int)
    if distance_type == 'euclidean':
        d = np.linalg.norm(x-x[0], axis=1)
        Rp = np.zeros((n, cp))
        Rp[:,0] = d
        for t in range(1, cp):
            d = np.minimum(d, Rp[:,t-1])
            m[t] = np.argmax(d)
            np.subtract(x[m[t]], x, out=Rp[:, t])
            np.linalg.norm(Rp[:, t], ord=2, axis=1, out=Rp[:, t])
    elif distance_type == 'cosine':
        d = 1 - x.dot(x[0]) / (np.linalg.norm(x, axis=1) * np.linalg.norm(x[0]))
        Rp = np.ones((n, cp))
        Rp[:, 0] = d
        for t in range(1, cp):
            d = np.minimum(d, Rp[:, t-1])
            m[t] = np.argmax(d)
            Rp[:, t] = 1 - x.dot(x[m[t]]) / (np.linalg.norm(x, axis=1) * np.linalg.norm(x[m[t]]))
    else:
        raise ValueError("Invalid distance_type. Must be 'euclidean' or 'cosine'.")
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


def optimized_MMRS(x, cp, ns):
    """
    Perform MinMax Random Sampling on a data set.
    
    Parameters:
    x (numpy.ndarray): 2D array representing the dataset.
    cp (int): Integer representing the number of representative points to be selected.
    ns (float): The desired size of the final sample, as a proportion of the total data set.
    
    Returns:
    smp (numpy array): shape (sample_size,) A numpy array of indices representing the final sample of the data set.
    rp (numpy array): shape (n, cp) The responsibilities matrix resulting from the MM algorithm.
    m (numpy array): shape (cp, p) The cluster centroids resulting from the MM algorithm.
    """
    n, p = x.shape
    m, rp = MM(x, cp)
    i = np.argmin(rp, axis=1)
    smp = []
    for t in range(cp):
        s = np.where(i==t)[0]
        nt = int(np.ceil(ns*len(s)/n))
        ind = np.random.choice(s, size=nt, replace=False)
        smp.extend(ind)
    smp = np.array(smp)
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

def optimized_clusivat(x,cp,ns):
    """
    Perform MinMax Random Sampling on a data set.
    
    Parameters:
    x (numpy.ndarray): 2D array representing the dataset.
    cp (int): Integer representing the number of representative points to be selected.
    ns (float): The desired size of the final sample, as a proportion of the total data set.
    
    Returns:
    
    
    """
  
    smp,rp,m = MMRS(x,cp,ns)
    x_smp = x[smp]
    XX, YY = np.meshgrid(x_smp, x_smp)
    rs = np.linalg.norm(XX-YY, axis=-1)
    rv,C,I,ri,cut = iVAT(rs)
    return rv,C,I,ri,cut,smp
