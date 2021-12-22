;+
; NAME:
;   gspice_covar
;
; PURPOSE:
;   Compute covariance of cleaned data vector using BLAS matrix mult
;
; CALLING SEQUENCE:
;   out = gspice_covar(spec, /checkmean)
;
; INPUTS:
;   spec      - spectral data array (npix, nspec), must be float64
;
; KEYWORDS:
;   checkmean - set to check precision of mean subtraction
;
; OUTPUTS:
;   out       - list(cov, refmean)
;
; OPTIONAL OUTPUTS:
;   
; EXAMPLES:
;   
; COMMENTS:
;   All masking of spec must done before calling this.
;   
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
function gspice_covar, spec, checkmean=checkmean

  sz    = size(spec, /dimen)
  npix  = sz[0]
  nspec = sz[1]
  
; -------- check data type
  if size(spec, /tname) NE 'DOUBLE' then message, 'must pass float64'

; -------- make columns of Dmask mean zero for computation of covariance
  refmean = total(spec, 2)/nspec
  spec0   = spec  ; make a copy of spec
  for i=0L, npix-1 do spec0[i, *] -= refmean[i]

; -------- verify mean subtraction worked
  if keyword_set(checkmean) then begin 
     mnd = total(spec0, 2)/nspec
     print, 'Minmax mean data', minmax(mnd), stdev(mnd)
  endif 

; -------- compute covariance
  cov = matrixmultata(spec0)/(nspec-1) ; computes A' times A for intput A

  return, list(cov, refmean)
end


