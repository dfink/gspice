using Plots
using BenchmarkTools
using LinearAlgebra
using Statistics
push!(LOAD_PATH, pwd()*"/julia")
using gspice
using FITSIO


function gspice_unit_tests()

    println("Reading inputs...")
    f = FITS(pwd()*"/test/star-sample.fits.gz")
    spec = Float64.(read(f[2],"spec",case_sensitive=false)')
    ivar = Float64.(read(f[2],"ivar",case_sensitive=false)')
    mask = read(f[2],"mask",case_sensitive=false)'

    println("Reading IDL outputs...")
    testname = pwd()*"/test/gspice-unit-test-desi.fits.gz"
    fidl = FITS(testname)
    covmat_idl    = read(fidl[1])
    finalmask_idl = BitMatrix(read(fidl[2])')  # take transpose


    M  = covmat_idl
    Nd = size(M,1)

    # -------- test mkl_cholesky_invert()
    Minv = inv(cholesky(M))
    diff = std(Minv*M - I)
    println("mkl_cholesky_invert() : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end


    # -------- test gspice_submatrix_inv()
    imask = trues(Nd)
    imask[11:51] .= false

    Ainv1 = gspice_submatrix_inv(M, Minv, imask, bruteforce=true)
    Ainv2 = gspice_submatrix_inv(M, Minv, imask, bruteforce=false)

    diff = std(Ainv1-Ainv2)
    println("gspice_submatrix_inv() : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    # -------- test gspice_submatrix_inv_mult()
    Y = Float64.(spec[1:2, :]')
    MinvY = Minv*Y
    AinvY1 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=true)
    AinvY2 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=false)

    diff = std(AinvY1-AinvY2)/std(Y)
    println("gspice_submatrix_inv_mult() : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end


    # -------- test djs_maskinterp()

    mymask = [0.0  1.0  0.0  0.0  0.0  0.0;
            1.0  0.0  0.0  0.0  0.0  0.0;
            0.0  0.0  0.0  0.0  0.0  0.0]

    yval = [4.0  15.0   6.0  17.0   8.0  19.0;
            5.0  16.0   7.0  18.0   9.0  20.0;
            8.0  19.0  10.0  21.0  12.0  23.0]
    interp0 = djs_maskinterp(yval,mymask,axis=0,constant=true)
    interp0_correct = [4.0   5.0   6.0  17.0   8.0  19.0;
                       16.0  16.0   7.0  18.0   9.0  20.0;
                       8.0  19.0  10.0  21.0  12.0  23.0]
    if interp0 != interp0_correct throw("Failed") end

    interp1 = djs_maskinterp(yval,mymask,axis=1,constant=true)
    interp1_correct = [4.0  16.0   6.0  17.0   8.0  19.0;
                       6.0  16.0   7.0  18.0   9.0  20.0;
                       8.0  19.0  10.0  21.0  12.0  23.0]
    if interp1 != interp1_correct throw("Failed") end

    # -------- test gspice_standard_scale()
    Dvec, refscale, refmean = gspice_standard_scale(spec, ivar, mask)
    diff = (std(Dvec)/15.4904531385571)-1

    println("gspice_standard_scale() : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end


    # -------- test gspice_covar()
    naive_cov, __ = gspice_covar(spec, checkmean=true)


    diff = mean(naive_cov)/ 1095.4699138 - 1
    println("gspice_covar() naive_cov: ", diff)
    if abs(diff) > 1E-10 throw("Failed") end


    # -------- obtain the empirical covariance for this Dvec
    covmat, refmean = gspice_covar(Dvec)
    diff = (std(Dvec)/15.4904531385571)-1
    println("gspice_covar() check Dvec again : ", diff)
    diff = mean(Dvec)
    println("gspice_covar() check Dvec again : ", diff)

    # -------- gspice_gaussian_estimate()
    println(" ")

    icond = imask
    ipred = falses(Nd)
    covinv = inv(cholesky(covmat))

    ipred[31] = true
    predoverD0, predcovar, predkstar, kstar =
            gspice_gaussian_estimate(icond, ipred, covmat, Dvec, covinv; bruteforce=true)


    diff = (std(predkstar)/18.40340978594223)-1
    println("gspice_gaussian_estimate(brute): std(predkstar)  ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    diff = (predcovar/7.80784733148408)[1]-1
    println("gspice_gaussian_estimate(brute): predcovar  ",diff)
    if abs(diff) > 1E-8 throw("Failed") end

    predoverD0, predcovar, predkstar, kstar =
            gspice_gaussian_estimate(icond, ipred, covmat, Dvec, covinv; bruteforce=false)

    predkstar = Dvec*predoverD0
    diff = (std(predkstar)/18.40340978594223)-1
    println("gspice_gaussian_estimate(): std(predkstar)  ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    diff = (predcovar/7.80784733148408)[1]-1
    println("gspice_gaussian_estimate(): predcovar  ",diff)
    if abs(diff) > 1E-8 throw("Failed") end




    # -------- gspice_gp_interp
    pred, predvar = gspice_gp_interp(Dvec, covmat, nguard=20)

    diff = std(pred)/15.412908264176547 -1
    println("gspice_gp_interp(), pred : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    diff = std(predvar)/28.429495082182775-1
    println("gspice_gp_interp(), predvar : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    chi = (Dvec-pred)./sqrt.(predvar)
    diff = std(chi)/0.99990003600975130 -1
    println("gspice_gp_interp(), chi : ", diff)
    if abs(diff) > 1E-8 throw("Failed") end

    println("std(spec)",std(spec))
    println("std(ivar)",std(ivar))
    println("std(mask)",std(Float64.(mask)))




    # -------- gspice_chimask
    chimask = gspice_chimask(spec, ivar, (mask .!= 0), 20)

    diff = sum(chimask) - 444
    println("gspice_chimask() : ", diff)
    if abs(diff) > 1E-10 throw("Failed") end


    # -------- end to end test, via gspice_covar_iter_mask()
    covmat, finalmask = gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20,8,6], maxbadpix=64)

    diff = std(covmat-covmat_idl)
    println("gspice covmat: ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

    diff = std(finalmask-finalmask_idl)
    println("gspice finalmask: ", diff)
    if abs(diff) > 1E-10 throw("Failed") end

  return
end
