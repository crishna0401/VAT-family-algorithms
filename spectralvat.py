import numpy as np
from visualclustering.VAT import VAT
from visualclustering.iVAT import iVAT
import networkx as nx

class specvat():
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.rs = self.dissimilarity()

    def vat(self):
        res = VAT(self.rs)
        return res

    def ivat(self):
        res = iVAT(self.rs)
        return res

    def dissimilarity(self):
        diag_mat = np.identity(self.data.number_of_nodes(),dtype=float)
        lap_mat = diag_mat - nx.laplacian_matrix(self.data).toarray()
        eig_values, eig_vecs = np.linalg.eigh(lap_mat)
        sort_indices = np.argsort(eig_values)[::-1][:self.k]
        eigvec_stack = eig_vecs[:, sort_indices]
        dist_matrix = np.linalg.norm(eigvec_stack[:, None] - eigvec_stack, axis=2)
        dist_matrix = dist_matrix / np.max(dist_matrix)
        return dist_matrix
