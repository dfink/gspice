import numpy as np
from .djs_maskinterp import maskinterp as djs_maskinterp
from .gspiceMain import gp_interp

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
    if mask is None:
        pixmask = (ivar == 0)
    else:
        pixmask = np.where(mask == 0, 0, 1) #scale mask to 0 or 1

    #interpolate over masked pixels in the spectral direction
    spec = djs_maskinterp(yval = spec, mask = pixmask, axis = 1)#, const = ).astype(np.float64) #dependent on pydl ##/const??
    
    #renormalize each spectra by sqrt(mean(ivar))
    wt = 1 - pixmask # set weight such that only good pixels contribute 
    meanivar = np.sum(ivar * wt, axis = 1)/np.sum(wt, axis = 1)
    refscale = np.sqrt(meanivar)

    spec *= refscale[:, np.newaxis] #rescale data as roughly data/sigma
    refmean = spec.mean(axis = 0)
    spec -= refmean.reshape(1, -1) 

    return spec, refscale, refmean

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
    spec -= refmean.reshape(1, -1) 

    #verify that mean subtraction worked
    if checkmean is False:
        mnd = spec.mean(axis = 0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std()}")
    
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
    spec = standard_scale(flux, ivar, mask)

    #obtain empirical covariance for this data array
    cov = covar(spec)[0] #cov is element 0 of this function

    #compute GSPICE predicted mean and covariance
    pred, predvar = gp_interp(spec, cov)

    #compute z-score 
    chi = (spec - pred)/np.sqrt(predvar)

    #clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, [1, 1, 1]).astype(flag.dtype)
    
    return chimask 

from time import time
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
    wmask = np.where(objmask) #index of good spectra
    nmask = len(wmask) #number of good spectra

    assert nmask <= npix, "Not enough good spectra"

    chimask = np.zeros(mask[wmask].shape)
    for sigma in nsigma: #loop over iteratively masking
        print(f"Pass nsigma = {sigma}")
        thismask = np.logical_or(chimask, (mask[wmask] != 0))
        chimask = get_chimask(flux = flux[wmask], ivar = ivar[wmask], mask = thismask, 
                            nsigma = sigma)
        print(f"Mean chimask = {np.mean(chimask)}")
        print(f"Time: {time()- t0} seconds.")
        t0 = time()

    finalmask = np.ones((nspec, npix))
    finalmask[wmask] = np.logical_or(thismask, chimask)

    spec = standard_scale(flux=flux[wmask], ivar= ivar[wmask], mask=finalmask[wmask])
    cov, _ = covar(spec)

    return cov, finalmask