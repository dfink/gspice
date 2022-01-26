import unittest
import numpy as np
from astropy.io import fits

from gspice import standard_scale

#import sample data for testing purposes
dat = fits.open("/n/home08/dfink/gspice/test/star-sample.fits.gz")
spec = dat[1].data['spec']
ivar = dat[1].data['ivar']
mask = dat[1].data['mask']
print(mask.min())
print(mask.max())
cov = fits.open("../../test/gspice-unit-test-desi.fits.gz")[0].data
covinv = np.linalg.inv(cov)
atol = 1e-11 #absolute tolerance for comparison

class TestStandardScale(unittest.TestCase):
    Dvec, _, _ = standard_scale(spec, ivar, mask)
    test_output = np.std(Dvec)
    print(test_output)
    expected_output = 15.4904531385571 

    np.testing.assert_allclose(actual = test_output,
                               desired = expected_output,
                               atol = atol)
