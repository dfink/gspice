;+
; NAME:
;   gspice_chimask
;
; PURPOSE:
;   Compute mask of outliers with respect to GSPICE posterior variance
;
; CALLING SEQUENCE:
;   chimask = gspice_chimask(flux, ivar, mask, nsigma=nsigma, gspice=gspice)
;
; INPUTS:
;   flux     - flux array (npix, nspec)
;   ivar     - inverse variance array (npix, nspec)
;   
; OPTIONAL INPUTS:
;   mask     - input mask (0=good).  If not passed, use ivar EQ 0
;   
; KEYWORDS:
;   nsigma   - number of sigma to cut at.
;   gspice   - if calling Python module, pass it as gspice
;
; OUTPUTS:
;   chimask  - mask of nonconformant pixels (0=good)
;
; EXAMPLES:
;   
; COMMENTS:
;   This routine standard scales the input so each spectrum has the same
;     mean(ivar), interpolates over bad pixels (from input mask), then 
;     computes a covariance. 
;   The Gaussian pixelwise estimate based on that covariance is compared
;     to the data, and a Z-score ("chi") is computed using the GSPICE
;     posterior variance. The output mask is based on that Zscore, and
;     set to 1 for abs(Z) > nsigma. 
;   Finally the mask is dilated by marking any pixel on either side of 
;     a bad pixel as also bad. 
;
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
function gspice_chimask, flux, ivar, mask, nsigma=nsigma, gspice=gspice

; -------- interpolate over masked pixels in the spectral direction
  Dvec = gspice_standard_scale(flux, ivar, mask)

; -------- obtain the empirical covariance for this Dvec
  cov = (gspice_covar(Dvec))[0]      ; cov is element 0 of returned list

; -------- compute GSPICE predicted mean and variance
  gspice_gp_interp, Dvec, cov, pred, predvar

; -------- compute "chi" (really the Z-score)
  chi = (Dvec-pred)/sqrt(predvar)

; -------- clip at nsigma, dilate the mask by 1 pixel
  nspec = (size(flux, /dim))[1]
  pad = bytarr(1, nspec)
  chim = [pad, abs(chi) GT nsigma, pad]
 
  chimask = chim OR shift(chim, 1, 0) OR shift(chim, -1, 0)

  return, chimask[1:-2, *]
end

