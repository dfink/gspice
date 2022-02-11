#from random import gauss
import unittest
import numpy as np
from astropy.io import fits

from gspice import standard_scale, covar, get_chimask, gaussian_estimate, gp_interp

#import sample data for testing purposes
dat = fits.open("/n/home08/dfink/gspice/test/star-sample.fits.gz")
spec = dat[1].data['spec'].astype(np.float64)
ivar = dat[1].data['ivar'].astype(np.float64)
mask = dat[1].data['mask']
cov = fits.open("../../test/gspice-unit-test-desi.fits.gz")[0].data
covinv = np.linalg.inv(cov)
atol = 1e-11 #additive tolerance for comparison
rtol = 1e-08 #multiplicative tolerance for comparison

# data product to be used for subsequent testing
Dvec, _, _ = standard_scale(spec, ivar, mask)
covmat, refmean = covar(Dvec)
naive_cov, refmean = covar(spec, checkmean=False)

class TestStandardScale(unittest.TestCase):
    def test_data(self):
        test_output = np.std(Dvec, ddof = 1)
        expected_output = 15.4904531385571 
        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output, 
                                   atol = atol, rtol = rtol)

class TestCovar(unittest.TestCase):
    def test_data(self):
        naive_cov, _ = covar(spec, checkmean=True)
        test_output = np.mean(naive_cov)
        expected_output = 1095.4699138
        print(test_output)
        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol, rtol = rtol)



class TestGaussianEstimate(unittest.TestCase):
    #arbitrary masking
    imask = np.ones(cov.shape[0])
    imask[11:51] = 0
    
    icond = imask
    ipred = np.zeros(cov.shape[0])
    
    _, predcovar, predkstar, kstar = gaussian_estimate(icond,
                                              ipred, covmat, Dvec, 
                                              covinv, bruteforce = False)

    def test_predkstar(self):
        test_output = self.predkstar
        expected_output = 18.40340978594223

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol, rtol = rtol)

    def test_predcovar(self):
        test_output = self.predcovar
        expected_output = 7.80784733148408

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                   atol = atol, rtol = rtol)
    
    def test_bruteforce(self):
        test_predcovar, test_predkstar, _  = gaussian_estimate(self.icond,
                                              self.ipred, covmat, Dvec, 
                                              covinv, bruteforce = True)

        # np.testing.assert_allclose(actual = test_predkstar,
        #                            desired = self.predkstar,
        #                            atol = atol)

        np.testing.assert_allclose(actual = test_predcovar,
                                   desired = self.predcovar,
                                   atol = atol, rtol = rtol)

"""class TestGPInterp(unittest.TestCase):

    pred, predvar = gp_interp(Dvec, covmat, nguard = 20)

    def test_pred(self):
        test_output = np.std(self.pred)
        expected_output = 15.412908264176547

        np.testing.assert_allclose(actual = test_output, 
                                   desired = expected_output,
                                   atol = atol)

    def test_predvar(self):
        test_output = np.std(self.predvar)
        expected_output = 28.429495082182775

        np.testing.assert_allclose(actual = test_output, 
                                   desired = expected_output,
                                   atol = atol)"""

"""class TestGetChimask(unittest.TestCase):
    def test_data(self):
        chimask = get_chimask(spec, ivar, mask != 0, 20)
        test_output = np.sum(chimask)
        expected_output = 444

        np.testing.assert_allclose(actual = test_output,
                                   desired = expected_output,
                                  atol = atol)"""
            
if __name__ == '__main__':
     unittest.main()