from python.matrixUtils import submatrix_inv_mult


def gaussian_estimate(icond, ipred, cov, Dvec, covinv):
    """
    Computes Gaussian conditional estimate

        Parameters:
            icond (np.array) : index list of pixels to condition on ??is it nspec x npix or flattened?
            ipred (np.array) : index list of pixels to predict ??
            cov (np.ndarray) npix X npix : covariance matrix
            Dvec (np.ndarray) nspec X npix : data vector 
            covinv (np.ndarray) npix X npix : inverse of cov, to speed up computation

        Returns:
            predcovar (np.ndarray) npix X npix : predicted covariance matrix
            predoverD (np.ndarray) ?? : matrix to multiply by Dvec to get predicted spectra
    """

    #single ??
    #nspec, npix = Dvec.shape

    k = icond.nonzero() #where good data
    nk = len(k) #number of good pixels

    kstar = ipred.nonzero() #where to predict data
    nkstar = len(kstar) #number of pixels to predict on

    cov_kkstar     = (cov[kstar, :])[:, k]   #  [nkstar, nk]
    cov_kstark     = cov_kkstar.T
    cov_kstarkstar = (cov[kstar, :])[:, kstar]

    #compute icov_kk (dot) cov_kstark using GSPICE
    Y = cov[:, kstar]
    Minvy = covinv.dot(Y)

    #Ainvy is icov_kk (dot) cov_kstarK
    #Ainvy0 is Ainvy zero-padded
    Ainvy0 = submatrix_inv_mult(M=cov, Minv=covinv,  ) ?!

    #predkstar = Dvec.dot(Ainvy0) #takes time; optional
    predoverD = Ainvy0

    #compute prediction covariance (See RW, Ch. 2)
    predcovar = cov_kstarkstar - Y.T @ Ainvy0

    return predcovar, predoverD 

from time import time
def gp_interp(nguard = 20, rang = None):
    """
    Computes pixelwise conditional prediction for each pixel (GSPICE routine)

        Parameters:
            nguard (int) : number of guard pixels around GCE pixel(s)
            rang (np.array) 2 X 1 : 2 element array specifying range for prediction pixels
        Returns:
    """
    
    t0 = time ()

    #shape of input spectra array
    nspec, npix = Dvec.shape

    #range of spectral bins to operate on
    if rang is not None: 
        assert rang.size != 2, "Range must have 2 elements."
        i0 = rang[0]
        i1 = rang[1]
    else:
        i0 = 0
        #i1 = npix - 1
        i1 = npix

    #allocate output arrays
    szpred = i1 - i0 + 1 ??# idl for loop counts last element or n - 1?
    predvar = np.zeros((szpred, nspec))
    predoverD = np.zeros((szpred, npix))

    #pre-compute inverse covariance
    covinv = mkl_cholesky_invert(cov) ?? #where is this?
    ones = np.ones(nspec + 1)

    #loop over pixels
    for i in range(i0, i1):
        #ipred == 1 for pixels to be predicted
        ipred = np.zeros(npix)
        ipred[i] = 1

        #condition on reference pixels specified by icond
        icond = ??

        gaussian_estimate() ??

        predoverD[:, kstar - i0] = predoverD0 ??
        j = ??
        predvar ??

        if ((i % 100) == 0):
            print(f"Iteration # {i} finished at {time()} seconds.")

    pred = Dvec.dot(predoverD)
    print(f"Matrix multiplication time: {time() - t0}")

    return pred, predvar 
