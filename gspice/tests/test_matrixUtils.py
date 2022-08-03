import unittest
import numpy as np
from astropy.io import fits

from gspice import cholesky_inv, submatrix_inv, submatrix_inv_mult

#import sample data for testing purposes
dat = fits.open("/n/home08/dfink/gspice/test/star-sample.fits.gz")
spec = dat[1].data['spec']
ivar = dat[1].data['ivar']
mask = dat[1].data['mask']
cov = fits.open("../../test/gspice-unit-test-desi.fits.gz")[0].data
covinv = np.linalg.inv(cov)

atol = 1e-11 #absolute tolerance for comparison

class TestCholesky(unittest.TestCase):
    def test_simple_matrix(self):
        """
        Test inversion of a 3 x 3 matrix
        """
        test_input = np.array([[2, 1, 0],
                         [1, 2, 1],
                         [0, 1, 2]
                        ])
        test_output = cholesky_inv(test_input)
        expected_output = (1/4) * np.array([[3, -2, 1],
                                             [-2, 4, -2],
                                             [1, -2, 3]
                                             ])
        
        np.testing.assert_allclose(actual = test_output, 
                                    desired = expected_output,
                                    )

    def test_sample_data(self):
        """
        Test inversion of mini dataset gspice/test/gspice-unit-test-desi.fits
        """
        test_cov_inv = cholesky_inv(cov)
        test_output = cov @ test_cov_inv #should be identity matrix up to precision
        expected_output = np.identity(cov.shape[0])
        
        np.testing.assert_allclose(actual = test_output, 
                                    desired = expected_output, 
                                    atol = 1e-11
                                    )

class TestSubmatrixInv(unittest.TestCase):
    def test_simple_matrix(self):
        """
        Test submatrix inversion of a 3 x 3 matrix where submatrix is 2 x 2
        """
        test_input = np.array([[2, 1, 0],
                              [1, 2, 1],
                              [0, 1, 2]])
        test_output = submatrix_inv(M = test_input, Minv = np.linalg.inv(test_input),
                                    imask = np.array([1, 1, 0]), bruteforce = False)
        expected_output = (1/3) * np.array([[2, -1], 
                                           [-1, 2]])

        np.testing.assert_allclose(actual = test_output, desired = expected_output)

    def test_no_mask_removal(self):
        """
        Test submatrix inversion of a 3 x 3 matrix where submatrix is full matrix
        """
        test_input = np.array([[2, 1, 0],
                              [1, 2, 1],
                              [0, 1, 2]])
        test_output = submatrix_inv(M = test_input, Minv = np.linalg.inv(test_input),
                                    imask = np.array([1, 1, 1]), bruteforce = False)
        expected_output = (1/4) * np.array([[3, -2, 1], 
                                           [-2, 4, -2],
                                           [1, -2, 3]])

        np.testing.assert_allclose(actual = test_output, desired = expected_output)

    def test_brute_force(self):
        """Compare submatrix inversion with and without brute force"""

        #arbitrary masking
        imask = np.ones(cov.shape[0])
        imask[11:51] = 0

        test_brute = submatrix_inv(cov, covinv, imask, bruteforce = True)
        test_no_brute = submatrix_inv(cov, covinv, imask, bruteforce = False)

        np.testing.assert_allclose(actual = test_no_brute, desired = test_brute,
                                  atol = atol)

class TestSubmatrixInvMult(unittest.TestCase):
    # def test_simple_matrix(self):
    #     N = 6
    #     np.random.seed(seed = 42)
    #     b = np.random.randint(-10, 10, size = (N, N))
        
    #     M = (b + b.T)/2
    #     Minv = np.linalg.inv(Minv)
    #     Y = np.random.randint(-10, 10, size = )

    
    def test_sample_data(self):
        #arbitrary masking
        imask = np.ones(cov.shape[0])
        imask[11:51] = 0
        Y = (spec[0:2, :].T).astype(np.float64) #Choose two spectra
        MinvY = covinv @ Y 
        
        test_brute = submatrix_inv_mult(M = cov, Minv = covinv, 
                                        imask =imask, Y = Y, 
                                        MinvY = MinvY, 
                                        bruteforce=True)
        test_no_brute = submatrix_inv_mult(cov, covinv, imask, Y, 
                                           MinvY, 
                                           pad = False, 
                                           bruteforce=False)

        np.testing.assert_allclose(actual = test_no_brute, 
                                   desired = test_brute,
                                   atol = atol)

    def test_pad(self):
        #arbitrary masking
        imask = np.ones(cov.shape[0])
        imask[11:51] = 0
        Y = (spec[0:2, :].T).astype(np.float64) #Choose two spectra
        MinvY = covinv @ Y 
        k = np.where(imask)[0]
        test_pad = submatrix_inv_mult(M = cov, Minv = covinv, 
                                        imask =imask, Y = Y, 
                                        MinvY = MinvY,
                                        pad = True,  
                                        bruteforce=False)
        test_no_pad = submatrix_inv_mult(cov, covinv, imask, Y, 
                                           MinvY, 
                                           pad = False, 
                                           bruteforce=False)
        np.testing.assert_allclose(actual = test_pad[k, :], 
                                   desired = 0.,
                                   atol = atol)

if __name__ == '__main__':
    unittest.main()