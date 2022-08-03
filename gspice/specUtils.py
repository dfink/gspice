import numpy as np
from .djs_maskinterp import maskinterp
#from .gspiceMain import gp_interp
from .matrixUtils import cholesky_inv, submatrix_inv_mult

from time import time 

def standard_scale(spec, ivar, mask = None):
    """
    Scale input data to have uniform variance per spectrum 

        Parameters:
            spec (np.ndarray): spec (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix); 
                               0 == good, default is ivar == 0

        Returns:
            spec (np.ndarray): scaled flux array (npsec, npix)
    """

    #if no mask is passed, use ivar == 0 as mask (setting infinity error as bad pixel)
    tt = time()
    if mask is None:
        pixmask = (ivar == 0)
    else:
        pixmask = np.where(mask == 0, 0, 1) #scale mask to 0 or 1
    print(f"time: {time()-tt}")
    
    #interpolate over masked pixels in the spectral direction
    Dvec = maskinterp(yval = spec, mask = pixmask, axis = 0)#, const = ).astype(np.float64) #dependent on pydl ##/const??
    
    #renormalize each spectra by sqrt(mean(ivar))
    wt = 1 - pixmask # set weight such that only good pixels contribute 
    meanivar = np.sum(ivar * wt, axis = 1)/np.sum(wt, axis = 1)
    refscale = np.sqrt(meanivar)
    Dvec = Dvec * refscale[:, np.newaxis] #rescale data as roughly data/sigma
    refmean = Dvec.mean(axis = 0)
    Dvec = Dvec - refmean.reshape(1, -1) 
    
    return Dvec, refscale, refmean

def covar(spec, checkmean = False): ##DONE
    """
    Compute covariance of cleaned data vector using BLAS 

        Parameters:
            spec (np.ndarray) nspec X npix: spectral data (nspec, npix); must be float64

        Returns:
            cov (np.ndarray) npix X npix: covariance matrix of the pixels
            refmean (np.array) 1 X nspec: reference mean of each spectra
    """

    nspec, npix = spec.shape

    #check data type
    assert spec.dtype == 'float64', "spec.dtype must be float64"

    #make columns of Dmask mean 0 for covariance computation
    refmean = spec.mean(axis = 0)
    spec = spec - refmean.reshape(1, -1) 

    #verify that mean subtraction worked
    if checkmean is True:
        mnd = spec.mean(axis = 0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std(ddof = 1)}")
    
    #compute covariance
    cov = (spec.T @ spec)/(nspec - 1)
    return cov, refmean 

from scipy.ndimage import binary_dilation as dilate
def get_chimask (flux, ivar, mask, nsigma): ##DONE
    """
    Compute mask of outliers with respect to GSPICE posterior covariance.
    
        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix) (0 == good)
            nsigma (int)     : threshold to clip data at 
        Returns:
            chimask (np.ndarray): mask of nonconformat pixels (0 == good)
    """
    
    #interporlate over masked pixels in the spectral direction
    Dvec, refscale, refmean = standard_scale(flux, ivar, mask)

    #obtain empirical covariance for this data array
    covmat, _ = covar(Dvec) 

    #compute GSPICE predicted mean and covariance
    pred, predvar = gp_interp(Dvec, covmat, nguard=20)

    #compute z-score 
    chi = (Dvec - pred)/np.sqrt(predvar)

    #clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    print(f'flag : {flag.shape}')
    chimask = dilate(flag, np.array([[1, 1, 1]])).astype(flag.dtype)
    
    return chimask 

from time import time
def gaussian_estimate(icond, ipred, cov, Dvec, covinv, bruteforce = False):
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

    #single = len()
    nspec, npix = Dvec.shape

    k = np.where(icond)[0] #where is good data
    nk = len(k) #number of good pixels
    #print('k :', k[:20])
    #print('nk: ', nk)
    kstar = np.where(ipred)[0]
    nkstar = len(kstar) #number of pixels to predict on
    #print('kstar: ', kstar)
    #print(f'nkstar : {nkstar}')
    #print('cov ', cov.std(ddof = 1))
    cov_kkstar = cov[np.ix_(kstar, k)]
    #print('cov_kkstar: ', cov_kkstar.shape)
    #print(f'cov_kkstar 9: {cov_kkstar[0][9]} 10: {cov_kkstar[0][10]}')
    cov_kstark     = cov_kkstar.T
    #print('cov_kstark: ', cov_kstark.shape)
    #print(f'cov_kstark 9: {cov_kstark[9][0]} 10: {cov_kstark[10][0]}')
    cov_kstarkstar = cov[np.ix_(kstar, kstar)]
    #print(f'covk*k*: {cov_kstarkstar.shape}')
    #print(f'k*k*: {cov_kstarkstar}')


    if bruteforce:
        cov_kk = cov[np.ix_(k, k)]

        #Cholesky inversion is faster than SVD here
        icov_kk = cholesky_inv(cov_kk)
        del cov_kk #save memory

        #compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - cov_kkstar @ icov_kk @ cov_kstark

         #if single: ?? why is this necessary?
        
        #compute icov_kk (dot) cov_kstark
        tmp = cov_kkstar @ icov_kk
        del icov_kk 

        if (nkstar == 1): #?? why is this necessary?
            tmp2 = np.zeros(npix)
            tmp2[k] = tmp 
            predkstar = Dvec @ tmp2 
        else:
            predkstar = Dvec[:, k] @ tmp.T #this is memory intensive
            print("Using memory intensive code...")

        return predcovar, predkstar, kstar #?? return values are different. deprecate bruteforce?
    else:
        #compute icov_kk (dot) cov_kstark using GSPICE
        Y = cov[:, kstar]
        #print(Y.shape)
        #print(Y[25][0])
        
        #print('tst ',covinv.std(ddof = 1))
        Minvy = covinv @ Y
        #print('tst ', Minvy.shape)
        #print('tst ', Minvy.std(ddof = 1))
        
        #Ainvy is icov_kk (dot) cov_kstarK
        #Ainvy0 is Ainvy zero-padded
        #print(cov.std(ddof = 1))
        #print(covinv.std(ddof = 1))
        #print(icond.std(ddof = 1))
        #print(Y.std(ddof = 1))
        #print(Minvy.std(ddof = 1))
        Ainvy0 = submatrix_inv_mult(M=cov, Minv=covinv, imask=icond, Y = Y,
                                    MinvY=Minvy, pad = True)
        #print('tst ', Ainvy0.shape)
        #print('tst ', Ainvy0.std(ddof = 1))
        #print('tst ', Ainvy0[4][0])
        #print("aaaaaa")
        predkstar = Dvec @Ainvy0 #takes time; optional
        #print('tst ', predkstar.shape)
        #print('tst predkstar', predkstar.std(ddof =1))
        predoverD = Ainvy0
        # print('tst ', predoverD.shape)
        # print('tst ', predoverD.std(ddof = 1))  ##--GOOD UP TO HERE--##

        #compute prediction covariance (See RW, Ch. 2)
        # print('tst ',Ainvy0.shape)
        # print('tst ',(Y.T).shape)
        # print('tst ',cov_kstarkstar.shape)
        # print('tst ', (Y.T @ Ainvy0)[0][0]) #precision up to 1e-8 with Julia
        # print('tst ', cov_kstarkstar)
        predcovar = cov_kstarkstar - Y.T @ Ainvy0 #precision up to 1e-8 with Julia
        #print('tst ', predcovar[0][0])
        return predoverD, predcovar, predkstar, kstar

from time import time
def gp_interp(spec, cov, nguard = 20, irange = None, bruteforce = False):
    """
    Computes pixelwise conditional prediction for each pixel (GSPICE routine)

        Parameters:
            spec (np.ndarray) nspec X npix : spectra matrix 
            cov (np.ndarray) npix X npix : pixel covariance matrix
            nguard (int) : number of guard pixels around GCE pixel(s)
            irange (np.array) 2 X 1 : 2 element array specifying range for prediction pixels
        
        Returns:
            pred (np.ndarray) nspec X npix : GSPICE predicted spectra mean
            predcovar (np.ndarray) npix X npix : GSPICE predicted posterior spectra variance
    """
    
    #shape of input spectra array
    nspec, npix = spec.shape
    #print('tst ', nspec)
    #print('tst ', npix)
    #check proper dimensionality
    assert cov.shape[0] == cov.shape[1], "covariance must be square matrix."
    assert npix == cov.shape[0], "spectra and covariance have incompatible dimensions."
    
    #range of spectral bins to operate on
    if irange is not None: 
        assert irange.size != 2, "Range must have 2 elements."
        i0 = irange[0]
        i1 = irange[1]
    else:
        i0 = 0
        i1 = npix - 1
    #print('tst ', i0)
    #print('tst ', i1)
    
    #allocate output arrays
    szpred = i1 - i0 + 1
    #print('tst ', szpred) 
    predvar = np.zeros((nspec, szpred))
    #print('tst ', predvar)
    #print('tst ', predvar.shape) 
    
    if(bruteforce):
        pred = np.zeros((nspec, szpred))
    else:
        predoverD = np.zeros((npix, szpred))
        # print('tst ', predoverD)
        # print('tst ', predoverD.shape) 

    #t0 = time() #start timing

    #pre-compute inverse covariance
    covinv = cholesky_inv(cov)
    #print('tst ', covinv.std(ddof = 1))

    #loop over pixels
    for i in range(i0, i1 + 1):
        t0 = time() #start timing
        #ipred == 1 for pixels to be predicted
        ipred = np.zeros(npix)
        #print(ipred)
        #print(ipred.shape)
        ipred[i] = 1
        #print(ipred)
        #print(ipred.shape)

        #condition on reference pixels specified by icond
        icond = np.ones(npix)
        #print(icond)
        #print(icond.shape)  
        j0 = np.max([0, i - nguard + 1])
        j1 = np.min([i + nguard + 1, npix])
        #print(j0)
        #print(j1)
        icond[j0:j1] = 0
        #print(np.where(icond==0)) 

        #print(icond)
        #print(np.where(icond == 0))
        #print(len(np.where(icond == 0)[0]))

        #print('tst ', cov.std(ddof =1))
        #print(covinv.std(ddof = 1))
        predoverD0, predcovar, predkstar, kstar = gaussian_estimate(icond=icond, ipred=ipred, cov=cov,
                                                 Dvec=spec, covinv=covinv, bruteforce = bruteforce)
        #kstar = i
        #print(f'predoverD0 : {predoverD0.std(ddof = 1)}')
        #print(f'predcovar: {predcovar[0][0]}')
        #print(f'predcoar: {predcovar}')
        #print(f'predcoar: {predcovar.shape}') 

        if(bruteforce):
            pred[:, kstar - i0] = predkstar
        else:
            predoverD[:, kstar - i0] = predoverD0#[:,0] #?? #comparison with IDL and logic + diemnsinality of predoverD
            #print(predoverD.shape)
            #print(predoverD.std(ddof = 1))
        #j = np.arange(predcovar.ndim) #?? #this is getting dimensions?
        predvar[:, kstar - i0] = np.diag(predcovar)#[0,0] #?? #what is happening?
        #print(predvar.shape)
        #print(predvar.std(ddof =1))  ##GOOD UPTO HERE##

        if ((i % 100) == 0):
            print(f"Iteration # {i} finished at {time() - t0} seconds.")

    pred = spec.dot(predoverD)
    #print(pred.std(ddof = 1))
    #print(predvar.std(ddof = 1))
    print(f"Matrix multiplication time: {time() - t0}")

    return pred, predvar 

def covar_iter_mask(flux, ivar, mask, nsigma = np.array([20, 8, 6]), maxbadpix = 64):
    """
    Compute spectral covariance with iterative masking using GSPICE

        Parameters:
            flux (np.ndarray) nspec X npix : de-redshifted flux array
            ivar (np.ndarray) nspec X npix : de-redshifted ivar array
            mask (np.ndarray) nspec X npix : de-redshifted mask array, 0 == good;
                                             if not set, set to ivar == 0
            nsigma (np.array)              : array of nsigma clipping values to be passed to chimask
            maxbadpix (int)                : reject spectra with more than maxbadpix
                                             pixels masked in input mask      

        Returns:
            cov (np.ndarray) npix X npix : covariance of pixels
            finalmask (np.ndarray) nspec X npix : final mask after iteration 
    """

    t0 = time()

    nspec, npix = flux.shape

    #reject spectra with too many bad pixels
    nbadpix = np.sum(mask != 0, axis = 1) #number of bad pixels for each spectra
    objmask = nbadpix <= maxbadpix #flat for objects to be discarded (too many bad pix)
    wmask = np.where(objmask)[0] #index of good spectra
    nmask = len(wmask) #number of good spectra

    assert nmask >= npix, "Not enough good spectra"

    chimask = np.zeros(mask[wmask].shape)
    #thismask = np.logical_or(chimask, (mask[wmask] != 0))
    
    for sigma in nsigma: #loop over iteratively masking
        print(f"Pass nsigma = {sigma}")
        thismask = np.logical_or(chimask, (mask[wmask] != 0))
        
        chimask = get_chimask(flux = flux[wmask], ivar = ivar[wmask], mask = thismask, 
                            nsigma = sigma)
        print(f"Mean chimask = {np.mean(chimask)}")
        #thismask = np.logical_or(chimask, thismask)
        print(f"Time: {time()- t0} seconds.")
        t0 = time()

    finalmask = np.ones((nspec, npix))
    finalmask[wmask] = np.logical_or(thismask, chimask)

    spec, _, _ = standard_scale(spec=flux[wmask], ivar= ivar[wmask], mask=finalmask[wmask])
    cov, _ = covar(spec)

    return cov, finalmask
