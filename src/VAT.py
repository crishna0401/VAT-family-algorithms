import numpy as np

def VAT(R):
    """
    input: 
    R (n*n double): Dissimilarity data input
    outputs: 
    RV (n*n double): VAT-reordered dissimilarity data
    C (n int): Connection indexes of MST
    I (n int): Reordered indexes of R, the input data
    cut (n double): MST link cut magnitude
    """
    N, M = R.shape
    K = np.arange(N)
    J = K
    # P=zeros(1,N)
    y, i = np.max(R, axis=0), np.argmax(R, axis=0)
    y, j = np.max(y), np.argmax(y)
    I = i[j]
    J = np.delete(J, I)
    
    y, j = np.min(R[I, J]), np.argmin(R[I, J])
    I = np.append(I, J[j])
    J = np.delete(J, j)
    C = np.zeros(N, dtype=int)
    C[0:2] = 1
    cut = np.zeros(N)
    cut[1] = y
    for r in range(2, N-1):
        # y, i = np.min(R[I, J], axis=0), np.argmin(R[I, J], axis=0)
        y = np.zeros(len(J))
        i = np.zeros(len(J))
        for k in range(len(J)):
            y[k], i[k] = np.min(R[I, J[k]]), np.argmin(R[I, J[k]])

        y, j = np.min(y), np.argmin(y)
        I = np.append(I, J[j])
        J = np.delete(J, j)
        C[r] = i[j]
        cut[r] = y
    y, i = np.min(R[I, J], axis=0), np.argmin(R[I, J], axis=0)
    I = np.append(I, J)
    C[N-1] = i
    cut[N-1] = y
    RI = np.zeros(N, dtype=int)
    for r in range(N):
        RI[I[r]] = r
    RV = R[I, :]
    RV = RV[:, I]
    return RV, C, I, RI, cut


def optimized_VAT(R):
    """
    This function reorders dissimilarity data using VAT algorithm
    
    Parameters
    ----------
    R : ndarray
        dissimilarity data input (n*n double)
        
    Returns
    -------
    RV : ndarray
        VAT-reordered dissimilarity data (n*n double)
    C : ndarray
        Connection indexes of MST (n int)
    I : ndarray
        Reordered indexes of R, the input data (n int)
    cut : ndarray
        MST link cut magnitude (n double)
    """
    N, M = R.shape
    K = np.arange(N)
    J = K
    y, i = np.unravel_index(np.argmax(R), R.shape)
    j = np.argmin(R[i,:])
    I = [i, j]
    J = np.delete(J, [i, j])
    C = np.zeros(N)
    C[:2] = 1
    cut = np.zeros(N)
    cut[1] = R[i,j]

    for r in range(2, N-1):
        y, i = np.unravel_index(np.argmin(R[np.ix_(I, J)]), (len(I), len(J)))
        j = np.argmin(y)
        I.append(J[j])
        J = np.delete(J, j)
        C[r] = i+1
        cut[r] = R[I[i], J[j]]

    y, i = np.unravel_index(np.argmin(R[np.ix_(I, J)]), (len(I), len(J)))
    I.append(J[i])
    C[N-1] = i+1
    cut[N-1] = R[I[-2], J[i]]

    RI = np.zeros(N)
    for r in range(N):
        RI[I[r]] = r
    RV = R[np.ix_(I,I)]
    return RV, C, I, cut,RI
