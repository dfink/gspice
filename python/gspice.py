import numpy as np
import sys
from scipy import linalg
from scipy.linalg import lapack
from scipy import ndimage
from time import time as systime  # IDL function is "systime()"



# Invert a positive semi-definite matrix using Cholesky
def cholesky_inv(M):

    # Lower-triangular Cholesky factorization of M    
    L = np.linalg.cholesky(M)

    # Call LAPACK dpotri to get inverse (only lower triangle populated)    
    Minv0 = lapack.dpotri(L, lower=True)[0]

    # Copy lower triangle to upper triangle    
    Minv = Minv0+Minv0.T-np.diag(Minv0.diagonal())
    return Minv



# NAME:
#   submatrix_inv
#
# PURPOSE:
#   Given M and Minv, compute inverse of a submatrix of M
# 
# CALLING SEQUENCE:
#   Ainv = submatrix_inv(M, Minv, imask, bruteforce=False)
#
# INPUTS:
#   M      - (Nd,Nd) parent matrix, assumed symmetric and 
#            positive semi-definite.
#   Minv   - inverse of M, usually calculated with MKL Cholesky routines.
#            this must be passed.
#   imask  - mask of rows/columns to use (1=keep, 0=remove).  This
#            vector contains Nk ones and Nr zeros. 
#   
# OUTPUTS:
#   Ainv   - Inverse of the submatrix of M
#
# EXAMPLES:
#   See gspice routines
#   
# COMMENTS:
#   Let M by a block matrix
#
#          | A   B |                      | P   Q |
#     M  = |       |      and    M^{-1} = |       |
#          | B^T D |                      | Q^T U |
#
#   Then the inverse of submatrix A is the Schur complement of U,
#       
#     A^{-1} = P - Q U^{-1} Q^T
#
#   U and M must be invertible and positive semi-definite.
# 
# REVISION HISTORY:
#   2019-Oct-20 - Written by Douglas Finkbeiner, CfA   (At MPIA)
#   2019-Nov-11 - More compact form for A inverse
#   2020-May-27 - Translated to Python
#
#----------------------------------------------------------------------
def submatrix_inv(M, Minv, imask, bruteforce=False):

    # check dimensions of inputs
    Ndim = len(imask)
    Mdim = M.shape
    
    if (Mdim[0] != Mdim[1]):
        print('M must be a square matrix', file=sys.stderr)
        sys.exit()

    if (Ndim != Mdim[0]):
        print('M and imask have incompatible dimensions', file=sys.stderr)
        sys.exit()

    # rows/columns to keep (k) and remove (r)
    k    = np.array(imask) == 1
    nk   = np.sum(k)

    r    = np.array(imask) == 0
    nr   = np.sum(r)
    
    # brute force option, for testing
    if bruteforce:
        A    = (M[:, k])[k, :]
        Ainv = cholesky_inv(A)
        return Ainv

    # if there is nothing to do, return early
    if nr==0:
        print('imask does not remove any rows/columns...')
        return Minv

    Uinv = np.linalg.inv((Minv[r,:])[:,r])
    Qt   = (Minv[r, :])[:,k]
    Q    = Qt.T
    Ainv = (Minv[:, k])[k, :] - np.dot(Q, np.dot(Uinv, Qt))
    
    return Ainv




# NAME:
#   submatrix_inv_mult
#
# PURPOSE:
#   Given M, Minv, and MinvY, compute inverse of submatrix of M, times Y
# 
# CALLING SEQUENCE:
#   Ainvy = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, 
#           irange=, pad=, bruteforce=)
#
# INPUTS:
#   M      - (Nd,Nd) parent matrix, assumed symmetric and 
#            positive semi-definite
#   Minv   - inverse of M, usually calculated with MKL Cholesky routines.
#            this must be passed.
#   imask  - mask of rows/columns to use (1=keep, 0=remove).  This
#            vector contains Nk ones and Nr zeros. 
#   Y      - Matrix (Nspec,Nd) to multiply Ainv by (Nspec can be 1)
#   MinvY  - Minv times Y
#
#
# KEYWORDS:
#   irange - if imask=0 is contiguous, just give endpoints (faster)
#   pad    - zero-pad removed rows of Ainvy to Nd dimensions
#   bruteforce - brute force calculation for validation
#
# OUTPUTS:
#   Ainvy  - Inverse A (submatrix of M) times Y (Nspec, Nd-Nr)
#          - if pad is set, then zero-padded to (Nspec, Nd)
#             (with a zero at each removed row)
#
# EXAMPLES:
#   See gspice routines
#   
# COMMENTS:
#   Let M by a block matrix
#
#          | A   B |                      | P   Q |
#     M  = |       |      and    M^{-1} = |       |
#          | B^T D |                      | Q^T U |
#
#   Then the inverse of submatrix A is the Schur complement of U,
#       
#     A^{-1} = P - Q U^{-1} Q^T
#
#   U and M must be invertible and positive semi-definite.
#   This function returns 
#
#     A^{-1} Y = P Y - Q U^{-1} Q^T Y
#
#   Y is assumed to be zero-padded, i.e. Y[imask == 0]=0
#
# REVISION HISTORY:
#   2019-Oct-20 - Written by Douglas Finkbeiner, CfA   (At MPIA)
#   2019-Nov-11 - More compact form for A inverse
#   2020-May-27 - Translated to Python
#
# TODO:
#   implement irange keyword
#
#----------------------------------------------------------------------
def submatrix_inv_mult(M, Minv, imask, Y, MinvY, irange=1, pad=True, bruteforce=False):

    # check dimensions of inputs
    Ndim = len(imask)
    Mdim = M.shape
    
    if (Mdim[0] != Mdim[1]):
        print('M must be a square matrix', file=sys.stderr)
        sys.exit()

    if (Ndim != Mdim[0]):
        print('M and imask have incompatible dimensions', file=sys.stderr)
        sys.exit()

    if len(Y.shape) != 2:
        print('Y must be a column vector (or array)', file=sys.stderr)
        sys.exit()
        
    if len(MinvY.shape) != 2:
        print('MinvY must be a column vector (or array)', file=sys.stderr)
        sys.exit()
        

    # count rows/columns to keep (k) and remove (r)
    k    = np.array(imask) == 1
    nk   = np.sum(k)

    r    = np.array(imask) == 0
    nr   = np.sum(r)
    
    # brute force option (Slow - use only for testing!)
    if bruteforce:
        A     = (M[:, k])[k, :]
        Ainv  = cholesky_inv(A)
        Ainvy = np.dot(Ainv, Y[k, :])
        return Ainvy


    # if there is nothing to do, return early
    if nr==0:
        print('imask does not remove any rows/columns...')
        return MinvY


# -------- this is much faster if r indices are consecutive, pass irange
#     Use that Qty = Minv y - U y.  Qt is faster to gather than Q.

#  if keyword_set(irange) then begin 
#     U    = Minv[irange[0]:irange[1], irange[0]:irange[1]]
#     Yr   = Y[*, irange[0]:irange[1]]
#     Qty  = Minvy[*, irange[0]:irange[1]] - U ## Yr
#     Qt   = Minv[k, irange[0]:irange[1]]
#  endif else begin 
    U    = (Minv[r,:])[:, r]
    Yr   = Y[r, :]
    Qty  = MinvY[r, :] - np.dot(U, Yr)


# evaluate  A^{-1} Y = P Y - Q U^{-1} Q^T Y

# foolproof way
#    Uinv    = cholesky_inv(U)
#    UinvQtY = np.dot(Uinv, Qty)   

# Faster for big U (and fast enough for small U)
    if U.shape[0] == 1:
        UinvQtY = Qty/U[0]
    else:
        L = linalg.cho_factor(U, lower=False, check_finite=False)
        UinvQtY = linalg.cho_solve(L, Qty, overwrite_b=False)


    # -------- Evaluate   A^{-1} Y = P Y - Q U^{-1} Q^T Y
    # -------- with some shortcuts, using P Y = Minv Y - Q Y

    Qt   = (Minv[r,:])[:, k]   # This line takes most of the time ?!?!?


    if pad:
        AinvY0 = np.copy(MinvY)
        AinvY0[k, :] -= np.dot(Qt.T,UinvQtY+Yr)
        AinvY0[r, :] = 0
        return AinvY0

    AinvY = MinvY[k, :]- ( np.dot((UinvQtY+Yr).T, Qt) ).T

    return AinvY






def gaussian_estimate(wavekeep, wavemask, cov, Dvec, covinv, kstar=False, bruteforce=False, nomult=False):
# -------- wavemask is 1 for pixels to be predicted
# --------   conditional on the reference pixels specified by wavekeep

# -------- get index lists for reference (k) and interpolation inds
#          (kstar), following notation of RW Chapter 2 ~ Eq. 2.38

    single = len(Dvec.shape) == 1
    sz     = Dvec.shape

    k      = np.array(wavekeep) == 1       # where you have data
    nk     = np.sum(k)
    kstar  = np.array(wavemask) == 1       # where you want to interpolate
    nkstar = np.sum(kstar)



    if bruteforce:   # use old code
        cov_kk         = (cov[k, :])[:, k]
        cov_kkstar     = (cov[kstar, :])[:, k] #  dim [nk, nkstar] ????
        cov_kstark     = (cov[k, :])[:, kstar]
        cov_kstarkstar = (cov[kstar, :])[:, kstar]
        print(cov_kk)
# -------- Choleksy inversion is fine here and much faster than SVD. 
        icov_kk = cholesky_inv(cov_kk)
        cov_kk  = 0   # save memory

# -------- compute the prediction covariance (See RW, Chap. 2)
        print('cov_kstarkstar cov_kkstar icov_kk cov_kstark')
        print(cov_kstarkstar)
        print(cov_kkstar)
        print(icov_kk)
        print(cov_kstark)
        predcovar = cov_kstarkstar - (cov_kkstar.dot(np.dot(icov_kk,cov_kstark)))

        if single:       # multiple parts 2 and 3 first
            predkstar = cov_kkstar.dot(np.dot(icov_kk, Dvec[k].T))
        else:

# -------- compute icov_kk times cov_kstark 

            temp    = cov_kkstar.dot(icov_kk)
            icov_kk = 0
        
            if nkstar == 1:
                temp2     = np.zeros(sz[0])
                temp2[k]  = temp
                predkstar = Dvec.dot(temp2.T)
            else:
                predkstar = Dvec[k, :].dot(temp.T) # this takes memory ## Maybe not in Python?
                print('Using memory intensive code....')


    else:             # -------- GSPICE version

        cov_kkstar     = (cov[kstar, :])[:, k]   #  [nkstar, nk]
        cov_kstark     = (cov[k, :])[:, kstar]
        cov_kstarkstar = (cov[kstar, :])[:, kstar]

# -------- compute icov_kk times cov_kstark using GSPICE routine
     # could set Minvy for a slight speedup
        Y = cov[:, kstar]
        Minvy = covinv.dot(Y)


        Ainvy0 = submatrix_inv_mult(cov, covinv, wavekeep, Y, Minvy)
                                # Ainvy is icov_kk ## cov_kstark
                                # Ainvy0 is that zero padded

        predkstar = Dvec.dot(Ainvy0)  # this takes all the time. 
        predoverD = Ainvy0

# -------- compute the prediction covariance (See RW, Chap. 2)
#     predcovar = cov_kstarkstar - (cov_kkstar##(icov_kk##cov_kstark))
        predcovar = cov_kstarkstar - (Y.T).dot(Ainvy0)

    return predkstar, predcovar, predoverD





# Loop over pixels, do pixelwise prediction
def pixelwise_estimate(Dvec, cov, range=False, nguard=20):
    """
    Pixelwise Gaussian Conditional Estimation
    
    Parameters
    ----------
    Dvec : flux matrix
    cov : covariance matrix 
    rang : 
    nguard : size of guard window
    
    Returns
    -------
    pred : GSPICE flux matrix (same dim as Dvec)
    predvar : 
    """
    # Check inputs
    Ndim  = cov.shape
    if (Dvec.ndim == 1): Dvec=Dvec.reshape(1, len(Dvec))
    sz    = Dvec.shape
    npix  = sz[1]
    nspec = sz[0]
    print('pixelwise_estimate:  npix, nspec=', npix, nspec)
    
    if (Ndim[0] != Ndim[1]):
        print('cov must be a square matrix', file=sys.stderr)
        sys.exit()

    if (npix != Ndim[0]):
        print('Dvec and cov have incompatible dimensions', file=sys.stderr)
        sys.exit()

    # range of pixels to estimate
    if range:
        i0 = range[0]
        i1 = range[1]+1  # more Pythonic this way
    else:
        i0 = 0
        i1 = npix

    # allocate output arrays
    szpred = i1-i0
    predvar   = np.zeros((nspec, szpred))
    predoverD = np.zeros((npix,  szpred))

    t0 = systime()

    # -------- compute inverse covariance
    covinv  = cholesky_inv(cov)

    # -------- loop over spectral pixels
    for i in np.arange(i0, i1):

        # -------- wavemask is 1 for pixels to be predicted ...
        wavemask = np.zeros(npix, dtype=np.uint8)
        wavemask[i] = 1

        # -------- conditional on the reference pixels specified by wavekeep
        wavekeep = np.ones(npix, dtype=np.uint8)
        j0 = np.max([0,i-nguard])
        j1 = np.min([i+nguard+1,npix])
        wavekeep[j0:j1] = 0

        # -------- call the Gaussian conditional estimate code
        result = gaussian_estimate(wavekeep, wavemask, cov, Dvec, covinv, bruteforce=False, nomult=True)
        predcovar  = result[1]
        predoverD0 = result[2]
        kstar      = i

        predoverD[:, kstar-i0] = predoverD0[:,0]
#        j = np.arange(predcovar.shape[0])
        predvar[:, kstar-i0] = predcovar[0, 0]

        #print(i, systime()-t0)

    # do the matrix multiplication by Dvec all in one step for efficiency
    pred = np.dot(Dvec, predoverD)

    print('Matrix mult:', systime()-t0)

    return pred, predvar

