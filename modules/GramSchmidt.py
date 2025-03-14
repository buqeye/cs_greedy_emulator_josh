# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)

###   ###   ###   imports   ###   ###   ###
import numpy as np
from numba import njit


@njit(fastmath=True)
def MGS(A_matrix, new_vector, atol=1e-8, rtol=1e-8):
    r"""Modified Gram-Schmidt function that orthonormalizes the one vector to the basis.
    NOTE: This function assumes that `A_matrix` is already orthonormalized!
    
    Modified Gram Schmidt:
    Takes in a basis with vectors in the columns as `A_matrix`, and a proposed new vector to be added 
    to the basis, `new_vector`. Returns an orthonormal basis with or without the proposed new vector 
    added depending on if it gets orthonormalized out.

    Parameters
    ----------
    A_matrix : matrix-like numpy array
        This should have basis vectors in the columns and must already be orthonormalized.
    new_vector : vector-like numpy array
        The proposed new vector to be added to A_matrix.
    atol : number (optional)
        The atol used to check if new_vector is orthonormalized out of the basis.
    rtol : number (optional)
        The rtol used to check if new_vector is orthonormalized out of the basis.

    Returns
    -------
    orthonormal_matrix : matrix-like numpy array
        An orthonormal matrix that either is the same as the input, `A_matrix`, or is `A_matrix` with 
        the added proposed new basis vector that has been orthonormalized to the rest of the matrix.
        """
    n = A_matrix.shape[1]
    orthonormal_matrix = np.column_stack((A_matrix, new_vector))
    
    for j in np.arange(n + 1):
        # # check to see if the new vector should be orthonormalized out
        if np.allclose(orthonormal_matrix[:, -1], 0., atol=atol, rtol=rtol):
            return A_matrix
        # follow a modified gram schmidt process, but only "looking" at the final basis vector
        elif j != n:
            orthonormal_matrix[:, n] -= np.vdot(orthonormal_matrix[:, j],
                                                orthonormal_matrix[:, n]) * orthonormal_matrix[:, j]
        # normalize the last element if we've made it here
        else:
            orthonormal_matrix[:, j] = orthonormal_matrix[:, j] / np.linalg.norm(orthonormal_matrix[:, j])
    
    return orthonormal_matrix


@njit(fastmath=True)
def unoptimized_MGS(A_matrix):
    r"""Modified Gram-Schmidt function that orthonormalizes the _entire_ basis.
    
    Computes an orthonormal basis from the given basis vectors using modified gram schmidt. This is 
    the "unoptimized MGS" because this function does not make any assumptions to accelerate computation.
    
    Parameters
    ----------
    A_matrix : array like
        The basis stored in the columns of this matrix.

    Returns
    -------
    orthonormal_matrix : numpy array
        The orthonormalized basis given in `A_matrix`.
    """
    A = np.copy(A_matrix)
    orthogonal_matrix = np.copy(A)
    orthonormal_matrix = np.zeros(A_matrix.shape)

    for j in np.arange(A.shape[1]):
        orthonormal_matrix[:, j] = orthogonal_matrix[:, j] / np.linalg.norm(orthogonal_matrix[:, j])

        for k in np.arange(j, A.shape[1]):
            orthogonal_matrix[:, k] -= np.vdot(orthonormal_matrix[:, j],
                                               orthogonal_matrix[:, k]) * orthonormal_matrix[:, j]

    return orthonormal_matrix
