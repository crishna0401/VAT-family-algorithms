import numpy as np
from visualclustering.VAT import VAT
from visualclustering.iVAT import iVAT
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import random
import statistics as stats

class specvat():
    def __init__(self, data, k, cp, ns,use_cosine=False):
        self.data = data
        self.n, self.p = data.shape
        self.k = k
        self.cp = cp
        self.ns = ns
        self.use_cosine = use_cosine
        self.smp = None
        self.rp = None
        self.m = None
        self.rs = self.dissimilarity()

    def compute_pred(self,cut,I,gt_clusters=None):
        self.clusters = gt_clusters
        self.smp = np.array(self.smp)
        ind = np.argsort(cut)[::-1]
        ind = np.sort(ind[:self.clusters-1])

        Pi = np.zeros(self.n)
        Pi[self.smp[I[:ind[0]]]] = 0
        Pi[self.smp[I[ind[-1]:]]] = self.clusters-1
        for k in range(1, self.clusters-1):
            Pi[self.smp[I[ind[k-1]:ind[k]]]] = k

        nsmp = np.setdiff1d(np.arange(self.n), self.smp)
        if self.use_cosine:
            r = euclidean_distances(self.data[self.smp], self.data[nsmp])
        else:
            r = euclidean_distances(self.data[self.smp], self.data[nsmp])
        s = np.argmin(r, axis=0)
        Pi[nsmp] = Pi[self.smp[s]]

        return Pi


    def ClusterRelabellingPA(self,PredictedLabel,lables):
        samples=len(lables)
        NoofK=len(np.unique(lables))
        cluster_matrix_mod=np.zeros(samples)
        length_partition=np.zeros(NoofK)
        for i2 in range(NoofK):
            length_partition[i2]=len(np.where(PredictedLabel==i2)[0])
        
        length_partition_sort = np.sort(length_partition)[::-1]
        length_partition_sort_idx = np.argsort(length_partition)[::-1]
        index_remaining = np.arange(NoofK)
        
        for i2 in range(NoofK):
            original_idx = length_partition_sort_idx[i2]
            partition = np.where(PredictedLabel==original_idx)[0]
            proposed_idx = stats.mode(lables[partition])
            if(np.sum(index_remaining==proposed_idx)!=0):
                # proposed_idx
                cluster_matrix_mod[PredictedLabel==original_idx]=proposed_idx
            else:
                # index_remaining[0]
                cluster_matrix_mod[PredictedLabel==original_idx]=index_remaining[0]
            index_remaining = index_remaining[index_remaining!=proposed_idx]
        
        PA=(samples-len(np.where((lables-(cluster_matrix_mod)!=0))[0]))/samples*100
        return PA,cluster_matrix_mod

    def compute_accuracy(self, Pi, gt):
        return self.ClusterRelabellingPA(Pi, gt)

 
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
        if self.use_cosine:
            d = cosine_distances(x[0].reshape(1, -1), x)[0]
        else:
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
            if self.use_cosine:
                Similarity = cosine_distances(self.data[smp], self.data[smp])
            else: 
                Similarity = euclidean_distances(self.data[smp], self.data[smp], squared=True)

            self.smp = smp
            self.rp = rp
            self.m = m
            
        else:
            if self.use_cosine:
                Similarity = cosine_distances(self.data, self.data)
            else: 
                Similarity = euclidean_distances(self.data, self.data, squared=True)

        Adjacent = self.myKNN(Similarity, K=10)
        degreeMatrix = np.sum(Adjacent, axis=1)
        laplacianMatrix = np.diag(degreeMatrix) - Adjacent
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))

        Laplacian = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

        eig_values, eig_vecs = np.linalg.eig(Laplacian)
        sort_indices = np.argsort(eig_values)[:self.k]
        eigvec_stack = eig_vecs[:, sort_indices]
        # dist_matrix = np.linalg.norm(eigvec_stack[:, None] - eigvec_stack, axis=2)
        if self.use_cosine:
            dist_matrix = cosine_distances(eigvec_stack, eigvec_stack)
        else:
            dist_matrix = euclidean_distances(eigvec_stack, eigvec_stack, squared=True)
        dist_matrix = dist_matrix / np.max(dist_matrix)

        return dist_matrix
