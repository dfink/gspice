import unittest
import numpy as np
from astropy.io import fits

from gspice import maskinterp

class TestMaskinterp(unittest.TestCase):
    """
    Test mask interpolation scheme of a simple data matrix
    """
    
    mymask = np.array([[0.0,  1.0,  0.0,  0.0,  0.0,  0.0,],
                       [1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                       [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]])

    yval = np.array([[4.0,  15.0,   6.0,  17.0,  8.0,  19.0],
                     [5.0,  16.0,   7.0,  18.0,  9.0,  20.0],
                     [8.0,  19.0,  10.0,  21.0,  12.0,  23.0]])

    def test_axis0(self):
        test_output = maskinterp(self.yval, self.mymask, axis=0)
        expected_output = np.array([[4.0,  5.0,  6.0,  17.0,  8.0,  19.0],
                                    [16.0, 16.0, 7.0,  18.0,  9.0,  20.0],
                                    [8.0,  19.0, 10.0, 21.0, 12.0,  23.0]])
        np.testing.assert_allclose(actual = test_output, desired = expected_output)

    def test_axis1(self):
        test_output = maskinterp(self.yval, self.mymask, axis=1)
        expected_output = np.array([[4.0,  16.0,   6.0,  17.0,   8.0,  19.0],
                                    [6.0,  16.0,   7.0,  18.0,   9.0,  20.0],
                                    [8.0,  19.0,  10.0,  21.0,  12.0,  23.0]])
        np.testing.assert_allclose(actual = test_output, desired = expected_output)

if __name__ == '__main__':
    unittest.main()