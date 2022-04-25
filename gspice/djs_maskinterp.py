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