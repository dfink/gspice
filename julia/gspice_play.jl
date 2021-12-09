using Plots
using BenchmarkTools
using LinearAlgebra
using Statistics
push!(LOAD_PATH, pwd()*"/julia")
using gspice


# using Profile
# using FFTW
#
# using Optim
# using Measures
# using Images, FileIO
#
# using SparseArrays
# using FITSIO
# using Distributions

    #
    #
    #     elif ndim == 3:
    #         if xval is None:
    #             if axis == 0:
    #                 for i in 1:sz[1]):
    #                     for j in 1:sz[2]):
    #                         ynew[i, j, :] = djs_maskinterp1(yval[i, j, :],
    #                                                         mask[i, j, :],
    #                                                         constant=constant)
    #             elif axis == 1:
    #                 for i in 1:sz[1]):
    #                     for j in 1:sz[3]):
    #                         ynew[i, :, j] = djs_maskinterp1(yval[i, :, j],
    #                                                         mask[i, :, j],
    #                                                         constant=constant)
    #             else:
    #                 for i in 1:sz[2]):
    #                     for j in 1:sz[3]):
    #                         ynew[:, i, j] = djs_maskinterp1(yval[:, i, j],
    #                                                         mask[:, i, j],
    #                                                         constant=constant)
    #         else:
    #             if axis == 0:
    #                 for i in 1:sz[1]):
    #                     for j in 1:sz[2]):
    #                         ynew[i, j, :] = djs_maskinterp1(yval[i, j, :],
    #                                                         mask[i, j, :],
    #                                                         xval=xval[i, j, :],
    #                                                         constant=constant)
    #             elif axis == 1:
    #                 for i in 1:sz[1]):
    #                     for j in 1:sz[3]):
    #                         ynew[i, :, j] = djs_maskinterp1(yval[i, :, j],
    #                                                         mask[i, :, j],
    #                                                         xval=xval[i, :, j],
    #                                                         constant=constant)
    #             else:
    #                 for i in 1:sz[2]):
    #                     for j in 1:sz[3]):
    #                         ynew[:, i, j] = djs_maskinterp1(yval[:, i, j],
    #                                                         mask[:, i, j],
    #                                                         xval=xval[:, i, j],
    #                                                         constant=constant)
    #     else:
    #         raise ValueError('Unsupported number of dimensions.')
    # return ynew


# Np=4000, < 0.5 ms on holyfink01

Np = 1000
Nspec = 1
T = Float64
xx = randn(T,Np,Np*5)
Y = randn(T,Np,Nspec)
M = xx*xx'
Minv = inv(M)
MinvY = Minv*Y
imask = trues(Np)
imask[960:1000].=false
gs = gspice_submatrix_inv(M, Minv, imask, bruteforce=false)

t1 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY; bruteforce=false)
t2 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY; bruteforce=true)
t3 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY; irange=960:1000,bruteforce=false)

spec = randn(5000,Np)
ivar = rand(5000,Np)

flux = spec
covmat = M
mask = zeros(size(flux))

using FITSIO
f = FITS("star-sample.fits.gz")
spec = read(f[2],"spec",case_sensitive=false)'
ivar = read(f[2],"ivar",case_sensitive=false)'
mask = read(f[2],"mask",case_sensitive=false)'


function gspice_unit_tests()

    println("Reading inputs...")
    f = FITS(pwd()*"/test/star-sample.fits.gz")
    spec = Float64.(read(f[2],"spec",case_sensitive=false)')
    ivar = Float64.(read(f[2],"ivar",case_sensitive=false)')
    mask = read(f[2],"mask",case_sensitive=false)'

    println("Reading IDL outputs...")
    testname = pwd()*"/test/gspice-unit-test-desi.fits"
    fidl = FITS(testname)
    covmat_idl    = read(fidl[1])
    finalmask_idl = BitMatrix(read(fidl[2])')  # take transpose


    M  = covmat_idl
    Nd = size(M,1)

    # -------- test mkl_cholesky_invert()
    Minv = inv(cholesky(M))
    diff = std(Minv*M - I)
    println("mkl_cholesky_invert() : ", diff)
    if diff > 1E-10 throw("Failed") end


    # -------- test gspice_submatrix_inv()
    imask = trues(Nd)
    imask[11:51] .= false

    Ainv1 = gspice_submatrix_inv(M, Minv, imask, bruteforce=true)
    Ainv2 = gspice_submatrix_inv(M, Minv, imask, bruteforce=false)

    diff = std(Ainv1-Ainv2)
    println("gspice_submatrix_inv() : ", diff)
    if diff > 1E-10 throw("Failed") end

    # -------- test gspice_submatrix_inv_mult()
    Y = Float64.(spec[1:2, :]')
    MinvY = Minv*Y
    AinvY1 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=true)
    AinvY2 = gspice_submatrix_inv_mult(M, Minv, imask, Y, MinvY, bruteforce=false)

    diff = std(AinvY1-AinvY2)/std(Y)
    println("gspice_submatrix_inv_mult() : ", diff)
    if diff > 1E-10 throw("Failed") end


    # -------- test gspice_standard_scale()
    Dvec = gspice_standard_scale(spec, ivar, mask)
    diff = (sum(Dvec)/303698042.57075346)-1
    println("gspice_standard_scale() : ", diff)
    if diff > 1E-10 throw("Failed") end


    # -------- test gspice_covar()
    naive_cov, __ = gspice_covar(Float64.(spec), checkmean=true)


    diff = mean(naive_cov)/ 1095.4699138 - 1
    println("gspice_standard_scale() : ", diff)
    if diff > 1E-10 throw("Failed") end


    # -------- end to end test, via gspice_covar_iter_mask()
    covmat, finalmask = gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20,8,6], maxbadpix=64)

    diff = std(covmat-covmat_idl)
    println("gspice covmat: ", diff)
    if diff > 1E-10 throw("Failed") end

    diff = std(finalmask-finalmask_idl)
    println("gspice finalmask: ", diff)
    if diff > 1E-10 throw("Failed") end

  return
end
















# start ds9 instance
using SAOImageDS9, XPA
function ds9(image;min=0,max=100)
    if isnothing(XPA.find(r"^DS9:")) throw("Must start DS9!") end
    SAOImageDS9.draw(image',min=min,max=max)
    return
end

function ds9(image::BitMatrix)
    if isnothing(XPA.find(r"^DS9:")) throw("Must start DS9!") end
    SAOImageDS9.draw(Int8.(image'),min=0,max=1)
    return
end


gspice_covar_iter_mask(spec, ivar, mask, nsigma=[20, 8, 6], maxbadpix=64)
