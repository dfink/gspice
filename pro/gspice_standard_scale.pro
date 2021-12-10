;+
; NAME:
;   gspice_standard_scale
;
; PURPOSE:
;   Scale the input data to have uniform variance per spectrum
;
; CALLING SEQUENCE:
;   Dvec = gspice_standard_scale(flux, ivar, mask, refscale=refscale)
;
; INPUTS:
;   flux     - flux array (npix, nspec)
;   ivar     - inverse variance array (npix, nspec)
;
; OPTIONAL INPUTS:
;   mask     - input mask (0=good).  If not passed, use ivar EQ 0
;   
; KEYWORDS:
;   refscale - factor each spectrum was multiplied by (nspec)
;   refmean  - mean of each rescaled wavelength bin (npix)
;   
; OUTPUTS:
;   Dvec     - scaled flux array (npix, nspec)
;
; EXAMPLES:
;   
; COMMENTS:
;   Use either mask NE 0 or ivar EQ 0 as pixel mask
;   Interpolate over bad pixels with djs_maskinterp()
;   Then compute mean of ivar over good pixels for each spectrum
;     and set refscale =s qrt(meanivar), rescale spectra.
;   This routine always outputs float64. 
;
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
function gspice_standard_scale, flux, ivar, mask, refscale=refscale, refmean=refmean

; -------- if no mask is passed, use ivar=0 as mask
  pixmask = keyword_set(mask) ? mask NE 0 : ivar EQ 0

; -------- interpolate over masked pixels in the spectral direction
  Dvec = djs_maskinterp(double(flux), pixmask, iaxis=0, /const)

; -------- renormalize each spectrum by sqrt(mean ivar)
  wt = 1-pixmask
  meanivar = total(ivar*wt, 1)/total(wt, 1)
  refscale = sqrt(meanivar)
  
  for i=0L, n_elements(meanivar)-1 do $
     Dvec[*, i] *= refscale[i] ; roughly data/sigma

  sz    = size(Dvec, /dimen)
  npix  = sz[0]
  nspec = sz[1]
  refmean = total(Dvec, 2)/nspec
  for i=0L, npix-1 do Dvec[i, *] -= refmean[i]

  return, Dvec
end
