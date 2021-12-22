;+
; NAME:
;   gspice_covar_iter_mask
;
; PURPOSE:
;   Compute spectral covariance with iterative masking using GSPICE
;   
; CALLING SEQUENCE:
;   out = gspice_covar_iter_mask(flux, ivar, mask, nsigma=, maxbadpix=)
;
; INPUTS:
;   flux     - flux array (npix, nspec)
;   ivar     - inverse variance array (npix, nspec)
;   mask     - input mask (0=good).  If not passed, use ivar EQ 0
;   
; OPTIONAL INPUTS:
;   
; KEYWORDS:
;   nsigma    - array of nsigma values for chimask
;   maxbadpix - reject objects with more than maxbadpix pixels 
;                 masked in input mask.
;
; OUTPUTS:
;   out       - list(cov, finalmask)
; OPTIONAL OUTPUTS:
;   
; EXAMPLES:
;   
; COMMENTS:
;   Iteratively mask estimates of the covariance matrix
;   Pass this routine de-redshifted spectra with mask of definitely bad pixels (1=bad)
;   
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
function gspice_covar_iter_mask, flux, ivar, mask, nsigma=nsigma, $
                                     maxbadpix=maxbadpix

  t0 = systime(1)                   ; start time

  if ~keyword_set(nsigma)    then nsigma = [20, 8, 6]
  if ~keyword_set(maxbadpix) then maxbadpix = 64
  sz    = size(flux, /dimen)
  npix  = sz[0]
  nspec = sz[1]

; -------- reject spectra with too many bad pixels
  nbadpix = total(mask NE 0, 1)     ; bad pixels for each spectrum
  objmask = nbadpix LE maxbadpix    ; 0=bad (too many pixels masked)
  wmask   = where(objmask, nmask)   ; index list of spectra to use

  if nmask LE npix then message, 'Not enough good spectra!'

  chimask = 0B
  for iter = 0, n_elements(nsigma)-1 do begin 
     print, '=========================  Pass ', byte(iter+1), ',   cut at Nsigma = ', nsigma[iter]
     thismask = chimask OR (mask[*, wmask] NE 0)
     chimask = gspice_chimask(flux[*, wmask], ivar[*, wmask], thismask, nsigma=nsigma[iter], gspice=gspice)
     print, 'mean chimask', mean(chimask)
     print, 'Time: ', systime(1)-t0, ' sec'
  endfor

  finalmask = bytarr(npix, nspec)+1B    ; start from original mask, overwrite mask for good spectra
  finalmask[*, wmask] = thismask OR chimask

  Dvec = gspice_standard_scale(flux[*, wmask], ivar[*, wmask], finalmask[*, wmask], refscale=refscale, refmean=refmean)
  cov = (gspice_covar(Dvec))[0]         ; cov is element 0 of returned list

  return, list(cov, finalmask)
end
