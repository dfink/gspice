import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from copy import deepcopy

def cholesky_inv(M):
    """
    Returns inverse of positive-definite matrix M using Cholesky decomposition
    """

    # Lower-triangular Cholesky factorization of M    
    L = linalg.cholesky(M, lower = True)

    # Call LAPACK dpotri to get inverse (only lower triangle populated)    
    Minv0 = lapack.dpotri(L, lower=True)[0]

    # Copy lower triangle to upper triangle    
    Minv = Minv0+Minv0.T-np.diag(Minv0.diagonal())
    return Minv

def submatrix_inv(M, Minv, imask, bruteforce = False):
    """
    Returns inverse of submatrix of M

        Parameters:
            M (np.ndarray) N x N: symmetric and positive semi-definite matrix
            Minv (np.ndarray) N x N: inverse of M
            imask (np.ndarray) N x N: mask of rows/columns to use 
                                      (1 == keep, 0 == remove); contains Nk ones and Nr zeros ?? Assume imask is symmetric
            bruteforce (bool) : flag for using bruteforce approach

        Returns:
            Ainv (np.ndarray): Inverse of A (submatrix of M)

        Comments:
            Let M be block matrix given by
                |A   B|              |P   Q| 
            M = |     | and M^{-1} = |     |
                |B.T D|              |Q.T U|

            Then inverse of A is Schur complement of U
            A^{-1} = P - Q U^{-1} Q.T

            U and M must be invertible and positive semi-definite.
    """
    
    #verify proper dimensionalities
    assert M.shape[0] == M.shape[1], "M must be a square matrix."
    assert imask.shape[0] == M.shape[0], "M and imask have incompatible dimensions."

    #ensure imask is boolean type because will use logical operators
    imask = imask.astype(bool) 

    #rows/columns to keep (k) and remove (r)
    k = np.where(imask)[0] 

    r = np.where(~imask)[0] 
    nr = len(r)

    if bruteforce:
        A = M[np.ix_(k, k)]
        Ainv = cholesky_inv(A)
        return Ainv
    
    if(nr == 0):
        print("imask does not remove any rows or columns.")
        return Minv

    #Evaluate  A^{-1} = P - Q U^{-1} Q.T
    Uinv = linalg.inv(Minv[np.ix_(r, r)])
    Q = Minv[np.ix_(k, r)]
    #Q = Qt.T
    Ainv = Minv[np.ix_(k, k)] - Q @ Uinv @ Q.T 

    return Ainv

##?? why not call submatrix_inv here?
def submatrix_inv_mult(M, Minv, imask, Y, MinvY, pad = True, bruteforce = False):
    """
    Returns (inverse of submatrix of M) * Y

        Parameters:
            M (np.ndarray) N x N: symmetric and positive semi-definite matrix
            Minv (np.ndarray) N x N: inverse of M
            imask (np.ndarray) N x N: mask of rows/columns to use 
                                      (1 == keep, 0 == remove); contains Nk ones and Nr zeros 
            Y (np.ndarray) Nspec x N:  matrix multiply Ainv by; assumed to be zero-padded
            MinvY (np.ndarray) N x Nspec:  matrix Minv * Y 
            pad (bool)        : flag for zero-padding
            bruteforce (bool) : flag for using bruteforce approach

        Returns:
            Ainvy (np.ndarray): Inverse of A (submatrix of M) times Y (nspec X N - Nr)
                                where Nr = number of removed rows.
                                - If pad is True, then zero-padded to nspec X N with 
                                0 at each removed row

        Comments:
            Let M be block matrix given by
                |A   B|              |P   Q| 
            M = |     | and M^{-1} = |     |
                |B.T D|              |Q.T U|

            Then inverse of A is Schur complement of U
            A^{-1} = P - Q U^{-1} Q.T

            U and M must be invertible and positive semi-definite.
            This returns,
            A^{-1} Y = P Y - (Q U^{-1} Q.T) Y

            Y is assumed to be zero-padded at bad rows    
    """

    #verify proper dimensionalities
    assert M.shape[0] == M.shape[1], "M must be a square matrix."
    assert imask.shape[0] == M.shape[0], "M and imask have incompatible dimensions."
    assert Y.ndim == 2, "Y must be column vector."
    assert MinvY.ndim == 2, "MinvY must be column vector."

    #rows/columns to keep (k) and remove (r)
    k = np.where(imask.any(axis = 1))[0] #?? assume imask symmetric
    nk = len(k)

    r = np.where(~imask.any(axis = 1))[0] #must convert to bool since ~ is bitwise complement
    nr = len(r)

    if bruteforce:
        A = (M[k,:])[:,k]
        Ainv = cholesky_inv(A)
        Ainvy = Ainv.T @ Y[k,:]
        return Ainvy
    
    if(nr == 0):
        print("imask does not remove any rows or columns.")
        return MinvY

    #Use Q.T y = Minv y - U y ?? why? known result?
    U    = (Minv[r,:])[:, r]
    Yr   = Y[r, :]
    Qty  = MinvY[r, :] - np.dot(U, Yr)
    Qt = (Minv[r,:])[:,k]

    #evaluate A^{-1} Y = P Y - Q U^{-1} Q.T Y

    #Faster for big U (and fast enough for small U)
    if(U.shape[0] == 1):
        UinvQtY = Qty/U[0]
    else:
        L = linalg.cho_factor(U, lower = False, check_finite = False)
        UinvQtY = linalg.cho_solve(L, Qty, overwrite_b = False)

    #Evaluate A^{-1} Y = P Y - Q U^{-1} Q^T Y
    #using P Y = Minv Y - Q Y

    if(pad):
        AinvY0 = deepcopy(MinvY)
        AinvY0[k, :] -= Qt.T @ (UinvQtY + Yr) 
        AinvY0[r, :] = 0
        return AinvY0
    else:
        AinvY = MinvY[k,:] - ((UinvQtY + Yr).T @ Qt).T
        return AinvY