"""
Written by Aaron Meisner.
From https://github.com/desihub/gfa_reduce/blob/master/py/gfa_reduce/analysis/djs_maskinterp.py
"""

import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as fits
import time

def maskinterp1(yval, mask):
# omitting xval arg (assume regular grid), omitting const kw arg
# (assume const=True behavior is desired)

    yval = np.array(yval)

    mask = mask.astype(int)

    bad = (mask != 0)
    if np.sum(bad) == 0:
        return yval

    good = (mask == 0)
    ngood = np.sum(good)
    if ngood == 0:
        return yval
    
    if np.sum(good) == 1:
        return yval*0 + yval[good][0]

    ynew = yval
    ny = len(yval)

    igood = (np.where(good))[0]
    ibad = (np.where(bad))[0]
    f = interp1d(igood, yval[igood], kind='linear', fill_value='extrapolate')

    yval[bad] = f(ibad)

    # do the /const part
    if igood[0] != 0:
        ynew[0:igood[0]] = ynew[igood[0]]
    if igood[ngood-1] != (ny-1):
        ynew[(igood[ngood-1]+1):ny] = ynew[igood[ngood-1]]

    return ynew

def maskinterp(yval, mask, axis):

    mask = mask.astype(int)

    sh_yval = yval.shape
    sh_mask = mask.shape
    
    assert(len(sh_yval) == 2)
    assert(len(sh_mask) == 2)

    assert((sh_yval[0] == sh_mask[0]) and (sh_yval[1] == sh_mask[1]))

    assert((axis == 0) or (axis == 1))

    wbad = (np.where(mask != 0))
    ynew = np.copy(yval) #to avoid yval pointer getting changed and stupid bugs popping up in the process because Python changes global variables with pointers!!

    if axis == 0:
        # the y coord values of rows that need some interpolation
        bad_stripe_indices = np.unique(wbad[0])
    else:
        # the x coord values of columns that need some interpolation
        bad_stripe_indices = np.unique(wbad[1])

    if len(bad_stripe_indices) == 0:
        return ynew

    for ind in bad_stripe_indices:
        if axis == 0:
            ynew[ind, :] = maskinterp1(ynew[ind, :], mask[ind, :])
        else:
            ynew[:, ind] = maskinterp1(ynew[:, ind], mask[:, ind])

    return ynew

def average_bilinear(yval, mask):
    int0 = maskinterp(yval, mask, 0)
    int1 = maskinterp(yval, mask, 1)
    interp = (int0 + int1)/2.0

    return interp

def djs_maskinterp1_doug(yval, mask, extrap_scheme = 'linear', xval=None):
    
    igood = np.where(mask == 0)[0]
    ngood = len(igood)
    ny    = len(yval)
    ynew  = np.copy(yval)

    if ngood == ny:
        return ynew   # if all good
    if ngood == 0:
        return ynew   # if none good
    #if ngood == 1: ##??##
    #    return ynew   # if only one is good (IDL behavior)

    if xval is None:
        ibad  = np.where(mask != 0)[0]
        itp = interp1d(igood, ynew[igood], kind=extrap_scheme, fill_value = 'extrapolate')
        ynew[ibad] = itp(ibad)
    else:
        ii    = np.argsort(xval)
        igood = np.where(mask[ii] == 0)[0]
        ibad  = np.where(mask[ii] != 0)[0]
        itp=interp1d(xval[ii[igood]], ynew[ii[igood]], kind=extrap_scheme, fill_value = 'extrapolate')
        ynew[ii[ibad]] = itp(xval[ii[ibad]])
    
    return ynew

# def djs_maskinterp_doug(yval, mask, xval = None, axis= None, constant=false):
#     @assert size(mask) == size(yval) "mask must have the same shape as yval."
#     if ~isnothing(xval)
#         @assert size(mask) == size(yval) "xval must have the same shape as yval."
#     end
#     sz = size(yval)
#     ndim = length(sz)
#     ext_scheme = constant ? Flat() : Line()

#     if ndim == 1
#         ynew = djs_maskinterp1(yval, mask, ext_scheme; xval=xval)
#     else
#         if isnothing(axis)
#             throw("Must set axis if yval has more than one dimension.")
#         end
#         if (axis < 0) || (axis > ndim-1) || (axis - round(axis)) != 0
#             throw("Invalid axis value.")
#         end
#         ynew = similar(yval)
#         if ndim == 2
#             if isnothing(xval)
#                 if axis == 0
#                     @views for i in 1:sz[1]
#                         ynew[i, :] = djs_maskinterp1(yval[i, :], mask[i, :],
#                                                      ext_scheme)
#                     end
#                 else
#                     @views for i in 1:sz[2]
#                         ynew[:, i] = djs_maskinterp1(yval[:, i], mask[:, i],
#                                                      ext_scheme)
#                     end
#                 end
#             else
#                 if axis == 0
#                     @views for i in 1:sz[1]
#                         ynew[i, :] = djs_maskinterp1(yval[i, :], mask[i, :], ext_scheme,
#                                                      xval=xval[i, :])
#                     end
#                 else
#                     @views for i in 1:sz[2]
#                         ynew[:, i] = djs_maskinterp1(yval[:, i], mask[:, i], ext_scheme,
#                                                      xval=xval[:, i])
#                     end
#                 end
#             end
#         end
#     end
#     return ynew
# end