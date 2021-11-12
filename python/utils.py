from numpy.lib.index_tricks import AxisConcatenator


from pydl.pydlutils.image import djs_maskinterp
def gspice_standard_scale(flux, ivar, mask = None)
    """
    Scale input data to have uniform variance per spectrum 

        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask : mask; 0 == good, default is ivar == 0

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
    wt = 1 - pixmask ##is this setting the opposite? why define weight this way??
    meanivar = np.sum(ivar * wt, axis = 1)/np.sum(wt, axis = 1)
    refscale = np.sqrt(meanivar)

    Dvec *= refscale[:, np.newaxis] #rescale data as roughly data/sigma

    return Dvec

def gspice_covar


from scipy.ndimage import binary_dilation as dilate
def gspice_chimask (flux, ivar, mask, nsigma = nsigma, gspice = gspice):
    """
    Compute mask of outliers with respect to GSPICE posterior covariance.
    
        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            
        Returns:
            chimask (np.ndarray): mask of nonconformat pixels (0 == good)
    """
    
    #interporlate over masked pixels in the spectral direction
    Dvec = gspice_standard_scale(flux, ivar, mask)

    #obtain empirical covariance for this data array
    cov = gspice_covar(Dvec)[0] #cov is element 0 of this function

    #compute GSPICE predicted mean and covariance
    pred, predvar = gspice_gp_interp(Dvec, cov)

    #compute z-score 
    chi = (Dvec - pred)/np.sqrt(predvar)

    #clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, [1, 1, 1]).astype(flag.dtype)
    
    return chimask 