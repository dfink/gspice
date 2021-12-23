import unittest
import numpy as np
from astropy.io import fits

from gspice import cholesky_inv

class TestCholesky(unittest.TestCase):
    def test_simple_matrix(self):
        """
        Test inversion of 3 x 3 matrix
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

        test_cov= fits.open("../../test/gspice-unit-test-desi.fits")[0].data
        test_cov_inv = cholesky_inv(test_cov)
        test_output = test_cov @ test_cov_inv #should be identity matrix up to precision
        expected_output = np.identity(test_cov.shape[0])
        
        np.testing.assert_allclose(actual = test_output, 
                                    desired = expected_output, 
                                    atol = 1e-11
                                    )

class TestSubmatrixInv(unittest.TestCase):
    

if __name__ == '__main__':
    unittest.main()