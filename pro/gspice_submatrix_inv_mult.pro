;+
; NAME:
;   gspice_submatrix_inv_mult
;
; PURPOSE:
;   Given M, Minv, and MinvY, compute inverse of submatrix of M, times Y
; 
; CALLING SEQUENCE:
;   Ainvy = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, $
;           irange=, /pad, /bruteforce)
;
; INPUTS:
;   M      - (Nd,Nd) parent matrix, assumed symmetric and 
;            positive semi-definite
;   Minv   - inverse of M, usually calculated with MKL Cholesky routines.
;            this must be passed.
;   imask  - mask of rows/columns to use (1=keep, 0=remove).  This
;            vector contains Nk ones and Nr zeros. 
;   Y      - Matrix (Nspec,Nd) to multiply Ainv by (Nspec can be 1)
;   MinvY  - Minv times Y
;
;
; KEYWORDS:
;   irange - if imask=0 is contiguous, just give endpoints (faster)
;   pad    - zero-pad removed rows of Ainvy to Nd dimensions
;   bruteforce - brute force calculation for validation
;
; OUTPUTS:
;   Ainvy  - Inverse A (submatrix of M) times Y (Nspec, Nd-Nr)
;          - if /pad is set, then zero-padded to (Nspec, Nd)
;             (with a zero at each removed row)
;
; EXAMPLES:
;   See gspice routines
;   
; COMMENTS:
;   Let M by a block matrix
;
;          | A   B |                      | P   Q |
;     M  = |       |      and    M^{-1} = |       |
;          | B^T D |                      | Q^T U |
;
;   Then the inverse of submatrix A is the Schur complement of U,
;       
;     A^{-1} = P - Q U^{-1} Q^T
;
;   U and M must be invertible and positive semi-definite.
;   This function returns 
;
;     A^{-1} Y = P Y - Q U^{-1} Q^T Y
;
;   Y is assumed to be zero-padded, i.e. Y[where(~imask)]=0
;
;   "~" means "not" in IDL
; 
; REVISION HISTORY:
;   2019-Oct-20 - Written by Douglas Finkbeiner, CfA   (At MPIA)
;   2019-Nov-11 - More compact form for A inverse
;
;----------------------------------------------------------------------
function gspice_submatrix_inv_mult, M, Minv, imask, Y, MinvY, irange=irange, pad=pad, bruteforce=bruteforce


; -------- check inputs
  if ~keyword_set(M)     then message, 'Must pass M'
  if ~keyword_set(Minv)  then message, 'Must pass Minv'
  if ~keyword_set(imask) then message, 'Must pass imask'

; -------- catch the case where Y is a row vector
  if size(Y, /n_dim) NE 2     then message, 'Y must be a column vector'
  if size(MinvY, /n_dim) NE 2 then message, 'MinvY must be a column vector'


; -------- check dimensions of inputs
  Nd = n_elements(imask) 
  if Nd NE (size(M, /dimen))[0] then stop

; -------- brute force option (Slow - use only for testing!)
  if keyword_set(bruteforce) then begin 
     k   = where(imask, nk)
     A = (M[*, k])[k, *]
     Ainv = mkl_cholesky_invert(A)
     Ainvy = matrixmult(Ainv, Y[*, k])
     return, Ainvy
  endif

  if ~keyword_set(MinvY) then message, 'Must pass MinvY'

; -------- index list to remove
  r   = where(~imask, nr)

; -------- if there is nothing to do, return early
  if nr EQ 0 then begin
     message, 'imask does not remove any rows/columns...', /info
     return, MinvY
  endif 

  k    = where(imask, nk)

; -------- this is much faster if r indices are consecutive, pass irange
;     Use that Qty = Minv y - U y.  Qt is faster to gather than Q.
  if keyword_set(irange) then begin 
     U    = Minv[irange[0]:irange[1], irange[0]:irange[1]]
     Yr   = Y[*, irange[0]:irange[1]]
     Qty  = Minvy[*, irange[0]:irange[1]] - U ## Yr
     Qt   = Minv[k, irange[0]:irange[1]]
  endif else begin 
     U    = (Minv[*, r])[r, *]
     Yr   = Y[*, r]
     Qty  = Minvy[*, r] - U ## Yr
     Qt   = (Minv[*, r])[k, *]
  endelse
  
; -------- evaluate  A^{-1} Y = P Y - Q U^{-1} Q^T Y


;  Uinv = invert(U)  ; replace this with Cholesky
;  UinvQtY = matrixmult(Uinv, Qty)   

  if n_elements(U) EQ 1 then begin 
     UinvQtY = Qty/U[0]
  endif else begin 
     mkl_cholesky, U
     UinvQtY = la_cholsol(U, Qty)
  endelse 

; -------- Evaluate   A^{-1} Y = P Y - Q U^{-1} Q^T Y
; -------- with some shortcuts, using P Y = Minv Y - Q Y
  if keyword_set(pad) then begin 
     AinvY0 = MinvY
     AinvY0[*, k] -= transpose( matrixmult(transpose(UinvQtY+Yr), Qt) )
     AinvY0[*, r] = 0
     return, AinvY0
  endif

  AinvY = MinvY[*, k]- transpose( matrixmult(transpose(UinvQty+Yr), Qt) )
  
  return, AinvY
  
end
