#!/usr/bin/python3

import numpy as np
import cvxpy as cp
import scipy.sparse
from .utilities import find_li_vectors

""" Calculate the connes distance between vector states.

Given an operator D, a set alg_generators of algebra elements and
vectors v, w, we calculate sup_a |<v,av> - <w,aw>| : || [D,a] || <= 1
numerically using cvxpy."""


def find_distance(params, state1, state2):
    """Calculate the Connes distance between vector states v and w."""
    v, w = state1.vector, state2.vector
    problem = cvxsolve(params, v, w)
    return abs(problem.value)


def cvxsolve(params, v, w):
    """Solve the convex problem associated with the connes
    distance."""

    alg_gen_l = params.alg_gen_l
    d_alg_gen_l = params.d_alg_gen_l

    var_dims = len(alg_gen_l)
    x = cp.Variable(var_dims)

    s = sum((X*gen for X, gen in zip(x, alg_gen_l)))
    ds = sum((X*gen for X, gen in zip(x, d_alg_gen_l)))

    # equivalently:
    # idm = np.diag(np.ones(params.size))
    # c = cp.hstack([cp.vstack([idm, ds.T]), cp.vstack([ds, idm])])
    # constraint = [c >> 0]
    constraint = [cp.norm(ds, 2) <= 1]

    # split v, w into their real and imaginary parts, take v-w
    z = block_real_imag_state(v) - block_real_imag_state(w)

    problem = cp.Problem(cp.Minimize(cp.trace(z@s)), constraint)

    problem.solve(**params.cvxsolver_args)

    return problem


class DistanceParameters:
    """Parameters required by the cvx solver; everything but v and w
    goes here."""

    def __init__(self, alg_generators, D, cvxsolver_args={}, **kwargs):
        self.alg_gen_l, self.d_alg_gen_l = parse_alg_generators(D,
                                                                alg_generators)
        self.D = D
        self.complex_size = D.shape[0]
        self.x0 = np.ones(self.complex_size)
        self.cvxsolver_args = cvxsolver_args
        self.size = 2*D.shape[0]


def parse_alg_generators(D, alg_generators):
    """Turn a list of (sparse) complex algebra elements into their
    real-imaginary splitting, and additionally return the list of
    their commutators with D."""

    # init_shape = alg_generators[0].shape
    alg_gen_asnp = (mat.todense() for mat in alg_generators)
    # alg_gen_flattened = np.array([gen.flatten() for gen in alg_gen_asnp])
    # alg_gen_reduced = find_li_vectors(alg_gen_flattened)

    alg_gen_l = (block_real_imag(np.array(a)) for a in alg_gen_asnp)
    # alg_gen_l = (block_real_imag(np.reshape(a, init_shape)) for a in alg_gen_reduced)
    alg_gen_l = [scipy.sparse.coo_matrix(mat) for mat in alg_gen_l]

    dD = block_real_imag(D)
    d_alg_gen_l = (commutator(dD, a.todense()) for a in alg_gen_l)
    d_alg_gen_l = [scipy.sparse.coo_matrix(mat) for mat in d_alg_gen_l]

    return alg_gen_l, d_alg_gen_l


def commutator(array1, array2):
    """Return the commutator of two matrices"""

    return array1@array2 - array2@array1


def block_real_imag_state(vect):
    """Given a complex vector v, return the real matrix m such that
    <v, av> = tr(v@a), assuming a is selfadjoint and of the form
    block_real_imag(mat)."""

    vr = vect.real
    vi = vect.imag
    # the imaginary part of the entire expression vanishes because we
    # assume a.real + i a.imag is selfadjoint
    # <v+iw|a.real|v+iw> = <v|a.real|v> + <w|a.real|w> + <iw|a.real|v>
    # + <v|a.real|iw>
    rr = rankonematrix(vr, vr) + rankonematrix(vi, vi)
    # <v + iw|i a.imag|v + iw> = i <v|a.imag|v> - i <w|a.imag|w> -
    # <w|a.imag|v> + <v|a.imag|w>
    ii = - rankonematrix(vi, vr) + rankonematrix(vr, vi)
    # argument is of the form [[a.real, -a.imag],[a.imag, a.real]]

    # <v|a.real|v> + <w|a.real|w> = tr [[rr, 0], [0, rr]] * [[a.real,
    # a.imag], [a.imag, a.real]] / 2 [[00, ii], [-ii, 0]] * [[a.real,
    # -a.imag], [a.imag, a.real]] = [[ii a.imag, ii a.real], [-ii
    # a.real, a.imag ii]]

    # - <w|a.imag|v> + <v|a.imag|w> = tr [[0, ii], [-ii, 0]] *
    # [[a.real, -a.imag], [a.imag, a.real]] / 2
    m = np.block([[rr/2, ii/2], [- ii/2, rr/2]])
    return m


def rankonematrix(vect1, vect2):
    """Given v, w, return the matrix m such that <v,aw> = tr(m@a)."""
    return np.einsum('i,j', vect1, vect2)


def block_real_imag(mat):
    """Split a complex matrix into a real matrix of twice the dimension."""

    r = mat.real
    i = mat.imag
    m = np.block([[r, -i], [i, r]])
    return m
