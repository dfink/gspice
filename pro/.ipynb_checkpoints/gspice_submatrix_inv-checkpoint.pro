;+
; NAME:
;   gspice_submatrix_inv
;
; PURPOSE:
;   Given M and Minv, compute inverse of a submatrix of M
; 
; CALLING SEQUENCE:
;   Ainv = gspice_submatrix_inv(M, Minv, imask, bruteforce=bruteforce)
;
; INPUTS:
;   M      - (Nd,Nd) parent matrix, assumed symmetric and 
;            positive semi-definite.
;   Minv   - inverse of M, usually calculated with MKL Cholesky routines.
;            this must be passed.
;   imask  - mask of rows/columns to use (1=keep, 0=remove).  This
;            vector contains Nk ones and Nr zeros. 
;   
; OUTPUTS:
;   Ainv   - Inverse of the submatrix of M
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
; 
; REVISION HISTORY:
;   2019-Oct-20 - Written by Douglas Finkbeiner, CfA   (At MPIA)
;   2019-Nov-11 - More compact form for A inverse
;
;----------------------------------------------------------------------
function gspice_submatrix_inv, M, Minv, imask, bruteforce=bruteforce

; -------- check inputs
  if ~keyword_set(M)     then message, 'Must pass M'
  if ~keyword_set(Minv)  then message, 'Must pass Minv'
  if ~keyword_set(imask) then message, 'Must pass imask'

; -------- check dimensions of inputs
  Nd = n_elements(imask)
  if Nd NE (size(M, /dimen))[0] then stop

; -------- brute force option, for testing
  if keyword_set(bruteforce) then begin 
     k   = where(imask, nk)
     A = (M[*, k])[k, *]
     Ainv = mkl_cholesky_invert(A)
     return, Ainv
  endif

; -------- index list to remove
  r   = where(~imask, nr)

; -------- if there is nothing to do, return early
  if nr EQ 0 then begin
     message, 'imask does not remove any rows/columns...', /info
     Ainv = Minv
     return, Minv
  endif 

; -------- Evaluate  A^{-1} = P - Q U^{-1} Q^T

  k    = where(imask, nk)
  Uinv = invert((Minv[*,r])[r,*])
  Qt   = (Minv[*, r])[k, *]
  Q    = transpose(Qt)
  Ainv = (Minv[*, k])[k, *] - matrixmult(Q, matrixmult(Uinv, Qt))

  return, Ainv
end
