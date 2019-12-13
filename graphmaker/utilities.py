#!/usr/bin/python3


from numpy.linalg import svd
from numpy import dot, array


def find_li_vectors(mymatrix, tresh=1e-8):
    """Reduce matrix to linear independence by SVD."""

    u, s, vh = svd(mymatrix, full_matrices=True)

    possvals = array([val for val in s if abs(val) > tresh])
    myrank = len(possvals)

    # new_basis = vh[:myrank]  # in this basis, mymatrix = newu * possvals
    # that is, matrix@vh.T has only zeros in the last n-matrix columns

    newu = u[..., :myrank]

    return newu * possvals
