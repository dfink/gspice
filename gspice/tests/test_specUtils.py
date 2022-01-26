from random import gauss
import unittest
import numpy as np
from astropy.io import fits

from gspice import standard_scale, covar, get_chimask, gaussian_estimate

#import sample data for testing purposes
dat = fits.open("/n/home08/dfink/gspice/test/star-sample.fits.gz")
spec = dat[1].data['spec'].astype(np.float64)
ivar = dat[1].data['ivar'].astype(np.float64)
mask = dat[1].data['mask']
cov = fits.open("../../test/gspice-unit-test-desi.fits.gz")[0].data
covinv = np.linalg.inv(cov)
atol = 1e-11 #absolute tolerance for comparison

"""class TestStandardScale(unittest.TestCase):
    def test_data(self):
        Dvec, _, _ = standard_scale(spec, ivar, mask)
        test_output = np.std(Dvec)
        expected_output = 15.4904531385571 

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol)

class TestCovar(unittest.TestCase):
    def test_data(self):
        naive_cov, _ = covar(spec, checkmean=True)
        test_output = np.mean(naive_cov)
        expected_output = 1095.4699138

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol)"""

# Empirical covariance of this dataset
covmat, refmean = covar(spec)

class TestGaussianEstimate(unittest.TestCase):
    #arbitrary masking
    imask = np.ones(cov.shape[0])
    imask[11:51] = 0
    
    icond = imask
    ipred = np.zeros(cov.shape[0])
    Dvec, _, _ = standard_scale(spec, ivar, mask)

    _, predcovar, predkstar, kstar = gaussian_estimate(icond,
                                              ipred, covmat, Dvec, 
                                              covinv, bruteforce = False)

    """def test_predkstar(self):
        test_output = self.predkstar
        expected_output = 18.40340978594223

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol)

    def test_predcovar(self):
        test_output = self.predcovar
        expected_output = 7.80784733148408

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol)
    """
    def test_bruteforce(self):
        test_predcovar, test_predkstar, _  = gaussian_estimate(self.icond,
                                              self.ipred, covmat, self.Dvec, 
                                              covinv, bruteforce = True)

        # np.testing.assert_allclose(actual = test_predkstar,
        #                            desired = self.predkstar,
        #                            atol = atol)

        np.testing.assert_allclose(actual = test_predcovar,
                                   desired = self.predcovar,
                                   atol = atol)

# class TestGetChimask(unittest.TestCase):
#     def test_data(self):
#         chimask = get_chimask(spec, ivar, mask != 0, 20)
#         test_output = np.sum(chimask)
#         expected_output = 444

#         np.testing.assert_allclose(actual = test_output,
#                                    desired = expected_output,
#                                   atol = atol)


if __name__ == '__main__':
    unittest.main()