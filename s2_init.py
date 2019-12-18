#!/usr/bin/python3

import math
import numpy as np
import scipy.sparse

from twosphere.twosphere import TwoSphere

"""Setup GraphmakerData for the Dirac sphere of given size."""


def prepare_gmdata(spinorsize, inputbasename=None, nthreads=1,
                     pot_coupling=3, cvxsolver_args={"eps": 10**-2,
                                                     "solver": "SCS",
                                                     "verbose": False},
                     **kwargs):
    """Prepare arguments for a GraphmakerData object for the standard Dirac sphere
    of given rank."""
    sphere = TwoSphere(spinorsize)
    D = sphere.DiracS2
    gmargs = {}
    gmargs['coordinates'] = get_coordinates(sphere)
    gmargs['sq_coordinates'] = get_sq_coordinates(spinorsize)
    gmargs['D'] = D
    gmargs['alg_generators'] = get_spherical_harmonics(
        spinorsize, sphere, D)
    gmargs['pot_coupling'] = pot_coupling
    gmargs['cvxsolver_args'] = cvxsolver_args
    gmargs['nthreads'] = nthreads

    return gmargs


def get_Db(D):
    """Return the perturbed Dirac operator as found in [LG19a]."""
    spinorsize = D.shape[0]
    mylmax = l_max(spinorsize)
    B = get_B(D)
    # lmax odd -> -1/2, lmax ev -> +1/2
    c = ((mylmax + 1) % 2) - 1/2
    Db = D + c * B
    return Db


def get_coordinates(sphere):
    """Get the coordinate matrices x, y and z of S^2 as acting on the
    Dirac eigenspinors."""
    size = sphere.spinorsize
    P = sphere.P
    # abel says Y =2P-1
    # <2019-06-10 Mon> A: just checked all this against numerical
    # integration again, to be sure; it's OK up to signs.
    Y = 2*P-np.identity(2*size)
    # and Y= y_i sigma^i
    z = Y[size:, size:]
    # and then I know that y1, y2 are the off diagonal corners
    urblock = Y[size:, :size]
    ulblock = Y[:size, size:]
    x = 0.5 * (urblock + ulblock)
    y = 0.5 * 1j * (urblock - ulblock)
    return [x, y, z]


def cutoff_largermat(mat, target_dim):
    """Cut off a matrix of higher dimension along the spectrum of the
    Dirac operator."""
    largesize = mat.shape[0]
    mylmax = l_max(target_dim)
    theirlmax = l_max(largesize)
    offset = sum([eigen_dimension(l) for l in range(mylmax+1, theirlmax+1)])
    posdim = int(target_dim / 2)
    negcolumnindices = np.arange(0, posdim)
    poscolumnindices = np.arange(posdim+offset, largesize-offset)
    mycolumnindices = np.append(negcolumnindices, poscolumnindices)
    myindices = np.ix_(mycolumnindices, mycolumnindices)
    return mat[myindices]


def get_sq_coordinates(target_dim):
    """Calculate squared large coordinate matrices, then project to
    target_dim."""
    mylmax = l_max(target_dim) + 1
    offset = eigen_dimension(mylmax)
    posdim = int(target_dim / 2)
    newposdim = posdim + offset
    sphere = TwoSphere(2*newposdim)
    bigcoordinates = get_coordinates(sphere)
    mymats = [cutoff_largermat(mat@mat, target_dim) for mat in bigcoordinates]
    return mymats


def eigen_dimension(l):
    return 2*l


def l_max(dim):
    return math.ceil((-1+math.sqrt(1+2*dim))/2)


def split_real_imag(matrix):
    """Split matrix into selfadjoint and antiselfadjoint parts."""
    realmat = (matrix + matrix.conjugate().T)/2
    imagmat = (matrix - matrix.conjugate().T)/2
    return realmat, -1j*imagmat


def commutator_nonzero_p(mat1, mat2):
    """True if mat1 and mat2 do _not_ commute."""
    norm = scipy.sparse.linalg.norm(mat1@mat2 - mat2@mat1)
    return not np.isclose(norm, 0)


def get_spherical_harmonics(target_dim, sphere, D):
    """Obtain the matrices Y_{lm} as acting on the Dirac eigenspinors."""
    mylmax = l_max(target_dim)
    sphlmax = 2*mylmax
    myD = scipy.sparse.coo_matrix(D)
    lms = [(l, m) for l in range(sphlmax + 1) for m in range(-l, l+1)]
    sphs = [scipy.sparse.coo_matrix(sphere.SPHarm(*lm)) for lm in lms]
    coss = [split_real_imag(mat)[0] for mat in sphs]
    sins = [split_real_imag(mat)[1] for mat in sphs]
    realsphs = coss + sins
    realsphs = [mat for mat in realsphs if commutator_nonzero_p(mat, myD)]
    return realsphs


def get_B(D):
    evals = np.diag(D)
    positivity = evals > 0
    posevals = evals[positivity]
    # f[L] = sin \pi L
    # D|l...> = (l+1/2) |l...>
    b_posevals = np.array([math.cos(math.pi * x) for x in posevals])
    b_negevals = -b_posevals
    b_evals = np.concatenate((b_negevals, b_posevals))
    b = np.diag(b_evals)
    return b
