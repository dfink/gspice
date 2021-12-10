


pro sphtest
npix = 100000
nstar = 10000000

bstar = randomn(iseed,nstar, /double)  
lstar = randomn(iseed,nstar, /double)  
bpix = randomn(iseed,npix, /double)  
lpix = randomn(iseed,npix, /double)  
t0=systime(1)
spherematch, lstar,bstar,lpix,bpix,1.0/3600.d,m1,m2,d12 & t1=systime(1)
print, t1-t0
end

pro gspice_unit_tests

  print, 'Reading inputs...'
  a = mrdfits('./test/star-sample.fits.gz', 1)
  spec = double(a.spec)
  ivar = double(a.ivar)
  mask = a.mask

  print, 'Reading IDL outputs...'
  testname = './test/gspice-unit-test-desi.fits.gz'
  covmat_idl = readfits(testname)
  finalmask_idl = readfits(testname, ext=1)


  M = covmat_idl
  Nd = (size(M, /dim))[0]

; -------- test mkl_cholesky_invert()
  Minv = mkl_cholesky_invert(M)
  diff = stdev(Minv#M - identity(Nd)) 
  print, 'mkl_cholesky_invert() : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'


; -------- test gspice_submatrix_inv()
  imask = bytarr(Nd)+1B
  imask[10:50] = 0B

  Ainv1 = gspice_submatrix_inv(M, Minv, imask, bruteforce=1)
  Ainv2 = gspice_submatrix_inv(M, Minv, imask, bruteforce=0)

  diff = stdev(Ainv1-Ainv2)
  print, 'gspice_submatrix_inv() : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'

; -------- test gspice_submatrix_inv_mult()
  Y = transpose(double(spec[*, 0:1]))
  MinvY = Minv##Y
  AinvY1 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=1)
  AinvY2 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=0)

  diff = stdev(AinvY1-AinvY2)/stdev(Y)
  print, 'gspice_submatrix_inv_mult() : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'


; -------- test gspice_standard_scale()
  Dvec = gspice_standard_scale(spec, ivar, mask)
  diff = (stdev(Dvec)/15.4904531385571d)-1
  print, 'gspice_standard_scale() : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'
  diff = mean(Dvec)
  print, 'gspice_standard_scale() : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'


; -------- test gspice_covar()
  res = gspice_covar(double(spec), /checkmean)
  naive_cov = res[0]

  diff = mean(naive_cov)/ 1095.4699138D - 1
  print, 'gspice_covar() naive_cov : ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'


; -------- obtain the empirical covariance for this Dvec
    res = gspice_covar(Dvec)
    covmat = res[0]

    diff = (stdev(Dvec)/15.4904531385571d)-1
    print, 'gspice_standard_scale() : ', diff
    if abs(diff) GT 1E-10 then message, 'Failed'
    diff = mean(Dvec)
    print, 'gspice_standard_scale() : ', diff
    if abs(diff) GT 1E-10 then message, 'Failed'

; -------- gspice_gaussian_estimate()
    icond = imask
    ipred = bytarr(Nd)
    covinv = mkl_cholesky_invert(covmat)

    ipred[30] = 1B
    gspice_gaussian_estimate, icond, ipred, covmat, Dvec, predkstar, predcovar, covinv=covinv, kstar=kstar, /bruteforce, predoverD=predoverD0

    diff = (stdev(predkstar)/18.40340978594223d)-1
    print, "gspice_gaussian_estimate(brute): std(predkstar)", diff
    if abs(diff) GT 1E-10 then message, 'Failed'

    diff = (predcovar/7.80784733148408d)-1
    print, "gspice_gaussian_estimate(brute): predcovar",diff
    if abs(diff) GT 1E-8 then message, 'Failed'




    gspice_gaussian_estimate, icond, ipred, covmat, Dvec, predkstar, predcovar, covinv=covinv, kstar=kstar, predoverD=predoverD0

    print, "gspice_gaussian_estimate(): std(predoverD0)",stdev(predoverD0)
    predkstar = matrixmult(Dvec, predoverD0)

    diff = (stdev(predkstar)/18.40340978594223d)-1
    print, "gspice_gaussian_estimate(brute): std(predkstar)", diff
    if abs(diff) GT 1E-10 then message, 'Failed'

    diff = (predcovar/7.80784733148408d)-1
    print, "gspice_gaussian_estimate(brute): predcovar",diff
    if abs(diff) GT 1E-8 then message, 'Failed'



; -------- gspice_gp_interp
    print
    gspice_gp_interp, Dvec, covmat, pred, predvar, nguard=20

    diff = stdev(pred)/15.412908264176547d -1
    print, "gspice_gp_interp(), pred : ", diff
    if abs(diff) GT 1E-10 then message, 'Failed'

    diff = stdev(predvar)/28.429495082182775d -1
    print, "gspice_gp_interp(), predvar : ", diff
    if abs(diff) GT 1E-10 then message, 'Failed'

    chi = (Dvec-pred)/sqrt(predvar)
    diff = stdev(chi)/0.99990003600975130d -1
    print, "gspice_gp_interp(), chi : ", diff
    if abs(diff) GT 1E-8 then message, 'Failed'

    print, "stdev(spec)",stdev(spec)
    print, "stdev(ivar)",stdev(ivar)
    print, "stdev(mask)",stdev(double(mask))




; -------- end to end test, via gspice_covar_iter_mask()
  res = gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20,8,6], maxbadpix=64)
  covmat = res[0]
  finalmask = res[1]

stop

  diff = stdev(covmat-covmat_idl)
  print, 'gspice covmat: ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'

  diff = stdev(finalmask-finalmask_idl)
  print, 'gspice finalmask: ', diff
  if abs(diff) GT 1E-10 then message, 'Failed'

  return
end
