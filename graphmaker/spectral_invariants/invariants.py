#!/usr/bin/env python

import math
import mpmath
import numpy as np

"""Estimate spectral zeta residues as in A. B. Stern, “Finite-rank
approximations of spectral zeta residues,” Letters in Mathematical
Physics (2018), 10.1007/s11005-018-1117-5."""


def m_generic(L):
    """Generic estimator m for the time scales. Valid in every
    dimension, for every residue."""

    return math.log(L+1)


def eps(L, m=m_generic):
    """Epsilon controls the smallest time scale at which we trust the
    heat trace."""

    return m(L)*math.log(L+1)/(L+.001)


def f0(lamb, dim):
    """f0 estimates the leading residue:
    tr f0(epsilon D^2) = c_0 epsilon^{-s_0} + O(epsilon^{-s_{1}})"""

    gf = mpmath.gammainc(1-dim/2, a=1)
    ff = math.exp(-1-lamb)/(1+lamb)
    return ff / gf


def f1(lamb, dim):
    """f1 estimates the next-leading residue:
    tr f1(epsilon D^2) = c_1 epsilon^{-s_1} + O(epsilon^{-s_{2}})"""

    gf = mpmath.gammainc(2-dim/2, a=1)
    ff1 = 2**(dim/2)*math.exp(-1-2*lamb)/(1+2*lamb)
    ff2 = -math.exp(-1-lamb)/(1+2*lamb)
    return (ff1+ff2)/gf


def volume(evas, multiplicity=None, dim=None,  m=m_generic):
    """Estimate the volume associated to Dirac eigenvalues evas,
    in dimension d, with estimator m for the time scale."""
    if not dim:
        dim = dimension(evas, m)
    if not multiplicity:
        multiplicity = estimate_multiplicity(dim)
    evasq = [l**2 for l in evas]
    L = max(evasq)
    scaledevasq = [lamb*eps(L, m) for lamb in evasq]
    tot = sum([f0(lamb, dim) for lamb in scaledevasq])
    tot = tot*(eps(L, m)**(dim/2))
    return float((4*math.pi)**(dim/2)*tot / multiplicity)


def curvature(evas, multiplicity=None, dim=None, m=m_generic):
    """Estimate the curvature (the first nonleading zeta residue)
    associated to Dirac eigenvalues evas, in dimension d, with
    estimator m for the time scale."""
    if not dim:
        dim = dimension(evas, m)
    if not multiplicity:
        multiplicity = estimate_multiplicity(dim)
    L = max([lamb**2 for lamb in evas])
    tot = (sum([f1(lamb**2 * eps(L, m), dim) for lamb in evas])
           * eps(L, m)**(dim/2-1))
    return float((6*(4*math.pi)**(dim/2))*tot / multiplicity)


def estimate_multiplicity(dimension):
    int_dim = int(dimension)
    if int_dim % 2 == 0:
        return 2**(int_dim/2)
    else:
        return 2**((int_dim - 1)/2)


def specdim(evs, t):
    """Estimate the spectral dimension associated to Dirac eigenvalues
    evs, when the truncated heat kernel is considered accurate up to
    timescales t or larger."""

    sqevs = [(q**2).real for q in evs]
    numers = [q*math.exp(-t*q) for q in sqevs]
    denoms = [math.exp(-t*q) for q in sqevs]
    numer = sum(numers)
    denom = sum(denoms)
    ds = 2*t*numer/denom
    return ds


def dimension(evas, m=m_generic):
    """Estimate the spectral dimension associated to Dirac eigenvalues
    evs, with estimator m for the time scale."""

    L = np.amax([l**2 for l in evas])
    return specdim(evas, t=eps(L, m))
