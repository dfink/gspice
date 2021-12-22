from gspice.matrixUtils import cholesky_inv, submatrix_inv_mult
import numpy as np 

def gaussian_estimate(icond, ipred, cov, spec, covinv, bruteforce = False):
    """
    Computes Gaussian conditional estimate

        Parameters:
            icond (np.array) : index list of pixels to condition on ??is it nspec x npix or flattened?
            ipred (np.array) : index list of pixels to predict ??
            cov (np.ndarray) npix X npix    : covariance matrix
            spec (np.ndarray) nspec X npix  : data vector 
            covinv (np.ndarray) npix X npix : inverse of cov, 
                                              to speed up computation
            bruteforce (bool) : flag to use bruteforce code

        Returns:
            predcovar (np.ndarray) npix X npix  : predicted covariance matrix
            predoverD (np.ndarray) npix X kstar : matrix to multiply by spec to 
                                                  get predicted spectra; 
                                                  kstar refers to row(s) to be
                                                  predicted
    """

    #single ??
    #nspec, npix = spec.shape

    k = icond.nonzero() #where good data
    nk = len(k) #number of good pixels

    kstar = ipred.nonzero() #where to predict data
    nkstar = len(kstar) #number of pixels to predict on

    cov_kkstar     = (cov[kstar, :])[:, k]   #  [nkstar, nk]
    cov_kstark     = cov_kkstar.T
    cov_kstarkstar = (cov[kstar, :])[:, kstar]

    if bruteforce:
        cov_kk = (cov[k,:])[:,k]

        #Cholesky inversion is faster than SVD here
        icov_kk = cholesky_inv(cov_kk)
        del cov_kk #save memory

        #compute prediction covariance (See RW, Ch. 2)

        predcovar = cov_kstarkstar - cov_kkstar @ icov_kk @ cov_kstark

         #if single: ?? why is this necessary?
        
        #compute icov_kk (dot) cov_kstark
        tmp = cov_kkstar.dot(icov_kk)
        del icov_kk 

        #if nkstar == 1: ?? why is this necessary?
        #    tmp2 = np.zeros()
        predkstar = spec[:, k] @ tmp.T #this is memory intensive
        print("Using memory intensive code...")

        return predkstar, predcovar #?? return values are different. deprecate bruteforce?
    else:
        #compute icov_kk (dot) cov_kstark using GSPICE
        Y = cov[:, kstar]
        Minvy = covinv.dot(Y)

        #Ainvy is icov_kk (dot) cov_kstarK
        #Ainvy0 is Ainvy zero-padded
        Ainvy0 = submatrix_inv_mult(M=cov, Minv=covinv, imask=icond, Y = Y,
                                    MinvY=Minvy, pad = True)

        #predkstar = spec.dot(Ainvy0) #takes time; optional
        predoverD = Ainvy0

        #compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - Y.T @ Ainvy0

        return predcovar, predoverD 

from time import time
def gp_interp(spec, cov, nguard = 20, rang = None):
    """
    Computes pixelwise conditional prediction for each pixel (GSPICE routine)

        Parameters:
            spec (np.ndarray) nspec X npix : spectra matrix 
            cov (np.ndarray) npix X npix : pixel covariance matrix
            nguard (int) : number of guard pixels around GCE pixel(s)
            rang (np.array) 2 X 1 : 2 element array specifying range for prediction pixels
        
        Returns:
            pred (np.ndarray) nspec X npix : GSPICE predicted spectra mean
            predcovar (np.ndarray) npix X npix : GSPICE predicted posterior spectra variance
    """
    
    t0 = time ()

    #shape of input spectra array
    nspec, npix = spec.shape

    #check proper dimensionality
    assert cov.shape[0] == cov.shape[1], "covariance must be square matrix."
    assert npix == cov.shape[0], "spectra and covariance have incompatible dimensions."
    
    #range of spectral bins to operate on
    if rang is not None: 
        assert rang.size != 2, "Range must have 2 elements."
        i0 = rang[0]
        i1 = rang[1]
    else:
        i0 = 0
        i1 = npix

    #allocate output arrays
    szpred = i1 - i0 
    predvar = np.zeros((nspec, szpred))
    predoverD = np.zeros((npix, szpred))

    #pre-compute inverse covariance
    covinv = cholesky_inv(cov)

    #loop over pixels
    for i in range(i0, i1):
        #ipred == 1 for pixels to be predicted
        ipred = np.zeros(npix)
        ipred[i] = 1

        #condition on reference pixels specified by icond
        icond = np.ones(npix)
        j0 = np.max([0, i - nguard])
        j1 = np.min([i + nguard + 1, npix])
        icond[j0:j1] = 0

        predcovar, predoverD0 = gaussian_estimate(icond=icond, ipred=ipred, cov=cov,
                                                 spec=spec, covinv=covinv)
        kstar = i


        predoverD[:, kstar - i0] = predoverD0[:,0] #?? #comparison with IDL and logic + diemnsinality of predoverD
        #j = np.arange(predcovar.ndim) #?? #this is getting dimensions?
        predvar[:, kstar - i0] = predcovar[0,0] #?? #what is happening?

        if ((i % 100) == 0):
            print(f"Iteration # {i} finished at {time()} seconds.")

    pred = spec.dot(predoverD)
    print(f"Matrix multiplication time: {time() - t0}")

    return pred, predvar 
