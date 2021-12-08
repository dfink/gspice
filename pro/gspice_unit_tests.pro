pro gspice_unit_tests

  print, 'Reading inputs...'
  a = mrdfits('/n/fink2/dfink/desi/everest/somestars.fits.gz', 1)
  spec = a.spec
  ivar = a.ivar
  mask = a.mask

  print, 'Reading IDL outputs...'
  testname = '/n/fink2/dfink/desi/everest/gspice-unit-test-desi.fits'
  covmat_idl = readfits(testname)
  finalmask_idl = readfits(testname, ext=1)


  M = covmat_idl
  Nd = (size(M, /dim))[0]

; -------- test mkl_cholesky_invert()
  Minv = mkl_cholesky_invert(M)
  diff = stdev(Minv#M - identity(Nd)) 
  print, 'mkl_cholesky_invert() : ', diff
  if diff GT 1E-10 then message, 'Failed'


; -------- test gspice_submatrix_inv()
  imask = bytarr(Nd)+1B
  imask[10:50] = 0B

  Ainv1 = gspice_submatrix_inv(M, Minv, imask, bruteforce=1)
  Ainv2 = gspice_submatrix_inv(M, Minv, imask, bruteforce=0)

  diff = stdev(Ainv1-Ainv2)
  print, 'gspice_submatrix_inv() : ', diff
  if diff GT 1E-10 then message, 'Failed'

; -------- test gspice_submatrix_inv_mult()
  Y = transpose(double(spec[*, 0:1]))
  MinvY = Minv##Y
  AinvY1 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=1)
  AinvY2 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=0)

  diff = stdev(AinvY1-AinvY2)/stdev(Y)
  print, 'gspice_submatrix_inv_mult() : ', diff
  if diff GT 1E-10 then message, 'Failed'


; -------- test gspice_standard_scale()
  Dvec = gspice_standard_scale(spec, ivar, mask)
  diff = (total(Dvec)/303698040.3640D)-1
  print, 'gspice_standard_scale() : ', diff
  if diff GT 1E-10 then message, 'Failed'


; -------- test gspice_covar()
  res = gspice_covar(double(spec), /checkmean)
  naive_cov = res[0]

  diff = mean(naive_cov)/ 1095.4699138D - 1
  print, 'gspice_standard_scale() : ', diff
  if diff GT 1E-10 then message, 'Failed'


; -------- end to end test, via gspice_covar_iter_mask()
  res = gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20,8,6], maxbadpix=64)
  covmat = res[0]
  finalmask = res[1]

  diff = stdev(covmat-covmat_idl)
  print, 'gspice covmat: ', diff
  if diff GT 1E-10 then message, 'Failed'

  diff = stdev(finalmask-finalmask_idl)
  print, 'gspice finalmask: ', diff
  if diff GT 1E-10 then message, 'Failed'

  return
end
