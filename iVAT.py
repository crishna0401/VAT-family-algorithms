import numpy as np
from VAT import VAT

def iVAT(R, VATflag=False):
    """
    Input parameters:
    R (n*n double): dissimilarity data input
    VATflag (boolean): TRUE - R is VAT-reordered
    
    Output Values:
    RV (n*n double): VAT-reordered dissimilarity data
    RiV (n*n double): iVAT-transformed dissimilarity data
    """
    N = R.shape[0]
    reordering_mat = np.zeros(N, dtype=int)
    reordering_mat[0] = 1
    if VATflag:
        RV = R
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = np.arange(r)
            y, i = np.min(RV[r, 0:r]), np.argmin(RV[r, 0:r])
            reordering_mat[r] = i
            RiV[r, c] = y
            cnei = c[c != i]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[i, cnei])
            RiV[c, r] = RiV[r, c]
    else:
        RV, C, _,_,_ = VAT(R)
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = np.arange(r)
            reordering_mat[r] = C[r]
            RiV[r, c] = RV[r, C[r]]
            cnei = c[c != C[r]]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[C[r], cnei])
            RiV[c, r] = RiV[r, c]
    return RiV, RV, reordering_mat


def optimized_iVAT(R, VATflag=False):
    """
    This function reorders dissimilarity data using iVAT algorithm
    
    Parameters
    ----------
    R : ndarray
        dissimilarity data input (n*n double)
    VATflag : bool, optional
        TRUE - R is VAT-reordered
        
    Returns
    -------
    RV : ndarray
        VAT-reordered dissimilarity data (n*n double)
    RiV : ndarray
        iVAT-transformed dissimilarity data (n*n double)
    reordering_mat : ndarray
        reordering index (1,N)
    """
    N = R.shape[0]
    reordering_mat = np.zeros(N)
    reordering_mat[0] = 1
    if VATflag:
        RV = R
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = np.arange(r)
            y, i = np.min(RV[r, :r]), np.argmin(RV[r, :r])
            reordering_mat[r] = i
            RiV[r, c] = y
            cnei = c[c != i]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[i, cnei])
            RiV[c, r] = RiV[r, c].T
    else:
        RV, C, I, cut,RI = VAT(R)
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = np.arange(r)
            RiV[r, c] = RV[r, C[r]]
            cnei = c[c != C[r]]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[C[r], cnei])
            RiV[c, r] = RiV[r, c].T
    return RV, RiV, reordering_mat
