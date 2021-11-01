;+
; NAME:
;   gspice_gaussian_estimate
;
; PURPOSE:
;   Compute Gaussian conditional esimtate
;
; CALLING SEQUENCE:
;   gspice_gaussian_estimate, icond, ipred, cov, Dvec, predkstar, $
;     predcovar, kstar=, covinv=, bruteforce=, predoverD=
;
; INPUTS:
;   icond   - index list of pixels to condition on
;   ipred   - index list of pixels to predict
;   cov     - covariance matrix (npix,npix)
;   Dvec    - data vector (npix,nspec)
;   covinv  - inverse covariance (to avoid computing again)
; 
; OPTIONAL INPUTS:
;   
; KEYWORDS:
;   bruteforce - do simple brute force calculation for debugging
;   
; OUTPUTS:
;   predkstar - prediction  (bruteforce code only)
;   predcovar - predicted covariance
;   predoverD - matrix to multiply by Dvec to obtain result. 
;
; OPTIONAL OUTPUTS:
;   kstar     - return kstar index list for debugging
;
; EXAMPLES:
;   
; COMMENTS:
;   In normal operation, this runs without the /bruteforce keyword and
;     returns predoverD.  The desired result is predoverD*Dvec.  Doing 
;     that as one matrix multiplication is much faster than multiplying
;     each time in the loop. 
;
; REVISION HISTORY:
;   2021-Oct-21 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
pro gspice_gaussian_estimate, icond, ipred, cov, Dvec, predkstar, predcovar, kstar=kstar, covinv=covinv, bruteforce=bruteforce, predoverD=predoverD
; -------- ipred is 1 for pixels to be predicted

; -------- conditional on the reference pixels specified by icond

; -------- get index lists forxs reference (k) and interpolation inds
;          (kstar), following notation of RW Chapter 2 ~ Eq. 2.38

  single = size(Dvec, /n_dim) EQ 1
  sz = size(Dvec, /dimen)

  k     = where(icond, nk)        ; where you have data
  kstar = where(ipred, nkstar)    ; where you want to predict

  if keyword_set(bruteforce) then begin  ; use old code
     cov_kk         = (cov[*, k])[k, *]
     cov_kkstar     = (cov[*, kstar])[k, *] ; dim [nk, nkstar]
     cov_kstark     = (cov[*, k])[kstar, *]
     cov_kstarkstar = (cov[*, kstar])[kstar, *]

; -------- Choleksy inversion is fine here and much faster than SVD. 
     icov_kk = mkl_cholesky_invert(temporary(cov_kk))

; -------- compute the prediction covariance (See RW, Chap. 2)
     predcovar = cov_kstarkstar - (cov_kkstar##(icov_kk##cov_kstark))

     if single then begin       ; multiple parts 2 and 3 first
        predkstar = cov_kkstar ## (icov_kk ## transpose(Dvec[k]))
     endif else begin 

; -------- compute icov_kk ## cov_kstark using GSPICE routine

        temp = cov_kkstar ## icov_kk
        icov_kk = 0
        
        if nkstar EQ 1 then begin 
           temp2 = dblarr(sz[0])
           temp2[k] = temp
           predkstar = Dvec ## transpose(temp2)
        endif else begin 
           predkstar = Dvec[k, *] ## transpose(temp) ; this takes memory
           print, 'Using memory intensive code....'
        endelse
     endelse 
  endif else begin              ; -------- GSPICE version

     cov_kkstar     = (cov[*, kstar])[k, *] ; dim [nk, nkstar]
     cov_kstark     = (cov[kstar, *])[*, k]
     cov_kstarkstar = (cov[*, kstar])[kstar, *]

; -------- compute icov_kk ## cov_kstark using GSPICE routine
     ; could set Minvy for a slight speedup
     Y = cov[kstar, *]
     Minvy = matrixmult(covinv, Y)
;     Minvy = dblarr(1, n_elements(icond))
;     Minvy[kstar] = 1.0d

     Ainvy0 = gspice_submatrix_inv_mult(cov, covinv, icond, Y, Minvy, /pad)
                                ; Ainvy is icov_kk ## cov_kstark
                                ; Ainvy0 is that zero padded

;     predkstar = matrixmult(Dvec, Ainvy0)  ; this takes all the time. 
     predoverD = Ainvy0

; -------- compute the prediction covariance (See RW, Chap. 2)
;     predcovar = cov_kstarkstar - (cov_kkstar##(icov_kk##cov_kstark))
     predcovar = cov_kstarkstar - transpose(Y)##Ainvy0

  endelse
  return
end
