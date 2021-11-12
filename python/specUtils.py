import numpy as np
from numpy import ma

from pydl.pydlutils.image import djs_maskinterp
def standard_scale(flux, ivar, mask = None):
    """
    Scale input data to have uniform variance per spectrum 

        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix); 
                               0 == good, default is ivar == 0

        Returns:
            Dvec (np.ndarray): scaled flux array (npsec, npix)
    """

    #if no mask is passed, use ivar == 0 as mask (setting infinity error as bad pixel)
    if mask is None:
        pixmask = ivar == 0

    #interpolate over masked pixels in the spectral direction
    Dvec = djs_maskinterp(yval = flux, mask = pixmask, axis = 1, const = ).astype(np.float64) #dependent on pydl ##/const?

    ##--FIGURE OUT WHAT wt IS AND THE LOOP

    #renormalize each spectra by sqrt(mean(ivar))
    wt = 1 - pixmask ?? ##is this setting the opposite? why define weight this way??
    meanivar = np.sum(ivar * wt, axis = 1)/np.sum(wt, axis = 1)
    refscale = np.sqrt(meanivar)

    Dvec *= refscale[:, np.newaxis] #rescale data as roughly data/sigma

    return Dvec

def covar(spec, checkmean = None): ##DONE
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
    if checkmean is not None:
        mnd = spec.mean(axis = 0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std()}")
    
    #compute covariance
    cov = (spec.T @ spec)/(nspec - 1)

    return cov, refmean 

from scipy.ndimage import binary_dilation as dilate ??
def chimask (flux, ivar, mask, nsigma): ##DONE
    """
    Compute mask of outliers with respect to GSPICE posterior covariance.
    
        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            nsigma (int)     : threshold to clip data at 
        Returns:
            chimask (np.ndarray): mask of nonconformat pixels (0 == good)
    """
    
    #interporlate over masked pixels in the spectral direction
    Dvec = standard_scale(flux, ivar, mask)

    #obtain empirical covariance for this data array
    cov = covar(Dvec)[0] #cov is element 0 of this function

    #compute GSPICE predicted mean and covariance
    pred, predvar = gspice_gp_interp(Dvec, cov)

    #compute z-score 
    chi = (Dvec - pred)/np.sqrt(predvar)

    #clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, [1, 1, 1]).astype(flag.dtype) ??
    
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
    wmask = objmask[objmask != 0] #index of good spectra
    nmask = len(wmask) #number of good spectra

    assert nmask <= npix, "Not enough good spectra"

    #chimask = np.zeros(mask.shape)

    for sigma in nsigma: #loop over iteratively masking
        print(f"Pass nsigma = {sigma}")
        thismask = mask[wmask] != 0 ?? #isnt or going to set most liberal defn?
        chimask = chimask(flux = flux[wmask], ivar = ivar[wmask], thismask, 
                            nsigma = sigma)
        print(f"Mean chimask = {np.mean(chimask)}")
        print(f"Time: {time()- t0} seconds.")
        t0 = time()

    finalmask = np.ones((nspec, npix))
    finalmask[wmask] = chimask ??

    Dvec = standard_scale(flux=flux[wmask], ivar= ivar[wmask], mask=finalmask[wmask])
    cov, _ = covar(Dvec)

    return cov, finalmask