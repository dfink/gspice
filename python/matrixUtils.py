def submatrix_inv_mult(M, Minv, Y, MinvY):
    """
    Returns inverse of submatrix of M * Y

        Parameters:
            M (np.ndarray) N x N: symmetric and positive semi-definite matrix
            Minv (np.ndarray) N x N: inverse of M
            imask (np.ndarray) N x N: mask of rows/columns to use 
                                      (1 == keep, 0 == remove); contains Nk ones and Nr zeros 
            Y (np.ndarray) Nspec x N:  matrix multiply Ainv by; assumed to be zero-padded
            MinvY (np.ndarray) N x Nspec:  matrix Minv * Y 

        Returns:
            Ainvy (np.ndarray): 
    """

    #catch the case where Y is a row vector
    ??

    #check dimensions of inputs
    Nd = ??

    if bruteforce is not None:
        k = ?? #where is nk and nr set?

def submatrix_inv(M, Minv, imask):
    """
    Returns inverse of a submatrix of M

        Parameters:
            M
            Minv
            imask

        Returns:
            Ainv
    """

    ###extra###
        # rows/columns to keep (k) and remove (r)
    k    = np.array(imask) == 1
    nk   = np.sum(k)

    r    = np.array(imask) == 0
    nr   = np.sum(r)