import numpy as np

def iVAT(R, VATflag=False):
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
        RV, C = VAT(R)
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = np.arange(r)
            RiV[r, c] = RV[r, C[r]]
            cnei = c[c != C[r]]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[C[r], cnei])
            RiV[c, r] = RiV[r, c]
    return RiV, RV, reordering_mat