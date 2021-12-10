pro gspice_make_desi_test_data

  print, 'Reading inputs...'
  a = mrdfits('./test/star-sample.fits.gz', 1)
  spec = double(a.spec)
  ivar = double(a.ivar)
  mask = a.mask

  res = gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20,8,6], maxbadpix=64)
; 37 sec

  covmat = res[0]
  finalmask = res[1]
  outname = '~/gspice/test/gspice-unit-test-desi.fits'
  writefits, outname, covmat
  writefits, outname, finalmask, /append

  return
end
