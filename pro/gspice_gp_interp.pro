;+
; NAME:
;   gspice_gp_interp
;
; PURPOSE:
;   Perform GSPICE prediction of each pixel, conditional on others
;
; CALLING SEQUENCE:
;   gspice_gp_interp, Dvec, cov, pred, predvar, range=, nguard=, $
;                     bruteforce=, usepython=
;
; INPUTS:
;   Dvec       - input data vector, already cleaned and scaled
;   cov        - input covariance to use for GCE prediction
;   
; OPTIONAL INPUTS:
;   
; KEYWORDS:
;   range      - range of spectral indices to compute (default to all)
;   nguard     - number of guard pixels around GCE pixel
;   bruteforce - use bruteforce code for debugging
;   usepython  - call Python package for comparison
;
; OUTPUTS:
;   pred       - predicted posterior mean
;   predvar    - predicted posterior variance
;
; EXAMPLES:
;   
; COMMENTS:
;   Computes the Gaussian Conditional Estimate (GCE) of each pixel,
;     conditional on the other pixels except those within radius nguard.
;
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
pro gspice_gp_interp, Dvec, cov, pred, predvar, range=range, nguard=nguard, bruteforce=bruteforce, usepython=usepython

  if keyword_set(usepython) then begin 
     gspice = Python.import('gspice')
  endif 
  if ~keyword_set(nguard) then nguard=20   ; number of guard pixels
  if ~keyword_set(cov) then stop

; -------- shape of input array
  sz    = size(Dvec, /dimen)
  npix  = sz[0]
  nspec = sz[1]

; -------- range of spectral bins to operate on
  if keyword_set(range) then begin 
     if n_elements(range) NE 2 then message, 'range must have 2 elements'
     i0 = range[0]
     i1 = range[1]
  endif else begin 
     i0 = 0
     i1 = npix-1
  endelse 

; -------- allocate output arrays
  szpred    = (i1-i0)+1
  predvar   = dblarr(szpred, sz[1])
  predoverD = dblarr(szpred, npix)

  t0 = systime(1)

; -------- pre-compute inverse covariance
  covinv = mkl_cholesky_invert(cov)
  ones = (dblarr(sz[1])+1)
; -------- loop over spectral pixels
  for i=i0, i1 do begin 

; -------- ipred is 1 for pixels to be predicted
     ipred = bytarr(npix)
     ipred[i] = 1B

; -------- conditional on the reference pixels specified by icond
     icond = ~(smooth(float(ipred), 1+2*nguard, /edge_trunc) GT 1E-3)

     t3 = systime(1)

     if keyword_set(usepython) then begin 
        result = gspice.gaussian_estimate(icond, ipred, cov, covinv, Dvec, bruteforce=0, nomult=1)
        predcovar  = result[1]
        predoverD0 = result[2]
        kstar      = where(ipred)
     endif else begin 
        gspice_gaussian_estimate, icond, ipred, cov, Dvec, predkstar, predcovar, covinv=covinv, kstar=kstar, bruteforce=bruteforce, predoverD=predoverD0
     endelse 

     predoverD[kstar-i0, *] = predoverD0
     j = lindgen((size(predcovar, /dimen))[0])
     predvar[kstar-i0, *] = predcovar[j, j] # ones

     if (i mod 100) eq 0 then print, i, systime(1)-t0
  endfor
  pred = matrixmult(Dvec, predoverD)
  print, 'Matrix mult:', systime(1)-t0

  return
end
