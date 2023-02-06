import numpy as np
from visualclustering.VAT import VAT
from visualclustering.iVAT import iVAT
from sklearn.metrics.pairwise import euclidean_distances
import random

class specvat():
    def __init__(self, data, k, cp, ns):
        self.data = data
        self.k = k
        self.cp = cp
        self.ns = ns
        self.smp = None
        self.rp = None
        self.m = None
        self.rs = self.dissimilarity()
 
    def vat(self):
        res = VAT(self.rs)
        return res

    def ivat(self):
        res = iVAT(self.rs)
        return res

    def MM(self,x, cp):
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

    def MMRS(self,x, cp, ns):
        n, p = x.shape
        m, rp = self.MM(x, cp)
        i = np.argmin(rp, axis=1)
        smp = []
        for t in range(cp):
            s = np.where(i==t)[0]
            nt = (np.ceil(ns*len(s)/n)).astype('int')
            # randomly sample nt points from s
            ind = random.sample(range(len(s)), nt,)
            smp.append(s[ind])
        smp = [item for sublist in smp for item in sublist]
        smp = list(set(smp))
        return smp, rp, m


    def myKNN(self,S, K, sigma=1.0):
        N = len(S)
        A = np.zeros((N,N))

        for i in range(N):
            dist_with_index = zip(S[i], range(N))
            dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
            neighbours_id = [dist_with_index[m][1] for m in range(K+1)] # xi's k nearest neighbours

            for j in neighbours_id: # xj is xi's neighbour
                A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
                A[j][i] = A[i][j] # mutually
        return A


    def dissimilarity(self):
        if len(self.data) > self.ns:
            print("data size is greater than 500, so using smart sampling")
            # Sample data and obtain cluster information
            smp, rp, m = self.MMRS(self.data, self.cp, self.ns)
            # Compute pairwise distances between the sampled data
            Similarity = euclidean_distances(self.data[smp], self.data[smp], squared=True)
            self.smp = smp
            self.rp = rp
            self.m = m
        else:
            Similarity = euclidean_distances(self.data)

        Adjacent = self.myKNN(Similarity, K=10)
        degreeMatrix = np.sum(Adjacent, axis=1)
        laplacianMatrix = np.diag(degreeMatrix) - Adjacent
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))

        Laplacian = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

        eig_values, eig_vecs = np.linalg.eig(Laplacian)
        sort_indices = np.argsort(eig_values)[:self.k]
        eigvec_stack = eig_vecs[:, sort_indices]
        dist_matrix = np.linalg.norm(eigvec_stack[:, None] - eigvec_stack, axis=2)
        dist_matrix = dist_matrix / np.max(dist_matrix)


        return dist_matrix
