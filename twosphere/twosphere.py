#!/usr/bin/env python

import numpy as np
import math
from operator import itemgetter
import scipy
import scipy.special
import py3nj
from functools import partial

"""Generate the operator system spectral triple C(S^2), L^2(S^2, S),
D_{S^2} up to a spectral cutoff."""


def geteigenspinors(spinorsize):
    """Return <spinorsize> ordered eigenspinors |l, m, s> of D_{S^2}"""

    e = ()
    myl = 1/2
    # loop-and-a-half (TM)
    while True:
        for m in np.arange(1/2, myl+1, 1):
            for ms in [1, -1]:
                for s in [-1, 1]:
                    e += ((myl, m*ms, s),)
        myl += 1
        if len(e) >= spinorsize:
            break

    e = e[:spinorsize]
    eigenspinors = sorted(e, key=itemgetter(2, 0, 1))
    return eigenspinors

def JCoeff(l, m, s, lp, mp, sp):
    """Coefficients of the real structure J, excluding complex
    conjugation."""
    coeff = int((l == lp) & (m == -mp) & (s == sp))*1j*(-1)**(int(m-1/2))*s
    return coeff


def gen_acoeff(l, m, s, lp, mp, sp):
    if (l == lp) & (m + 1 == mp) & (s == -1*sp):
        return -1*math.sqrt((l + m + 1)*(l - m)) / (2*l*(l+1))
    elif (l + 1 == lp) & (m + 1 == mp) & (s == sp):
        return math.sqrt((l + m + 1)*(l + m + 2))/(2*(l+1))
    elif (l - 1 == lp) & (m + 1== mp) & (s == sp):
        return -1*math.sqrt((l - m)*(l - m - 1)) / (2*l)
    else:
        return 0.0


def gen_bcoeff(l, m, s, lp, mp, sp):
    if (l == lp) & (m == mp) & (s == -1*sp):
        return m/(2*l*(l+1))
    elif (l + 1 == lp) & (m == mp) & (s == sp):
        return math.sqrt((l - m + 1)*(l + m + 1))/(2*(l+1))
    elif (l - 1 == lp) & (m == mp) & (s == sp):
        return math.sqrt((l - m)*(l + m)) / (2*l)
    else:
        return 0.0


def diracS2Coeff(l, m, s, lp, mp, sp):
    return int((l == lp) & (m == mp) & (s == sp))*(l+1/2)*s


def GammaCoeff(l, m, s, lp, mp, sp):
    return int((l == lp) & (m == mp) & (s == -sp))*-1


def matFromCoeffs(coeffs, eigenspinors):
    """Create matrix, given a function <lp,mp,sp|l,m,s> -> num."""

    spinorsize = len(eigenspinors)
    vectorcoeffs = np.vectorize(lambda i, j: coeffs(*eigenspinors[i], *eigenspinors[j]))
    mat = np.fromfunction(vectorcoeffs, (spinorsize, spinorsize), dtype=int)
    return mat


def doubleMatrix(mat):
    """Lift matrix to H oplus H."""

    dims = mat.shape
    myZero = np.zeros(dims)
    ans = np.block([[mat, myZero], [myZero, mat]])
    return ans


def comm(mat1, mat2):
    return mat1@mat2 - mat2@mat1


def relTrace(mat, spinorsize):
    """Return the relative trace [[A,B],[C,D]]->A+D."""

    top = mat[:spinorsize, :spinorsize]
    bottom = mat[spinorsize:, spinorsize:]
    return 2*(top+bottom)


def harmSpinorPrefactor(l, m):
    """Return the prefactor of the harmonic spinors, see Gracia-Bondia."""
    sign = (-1)**(l-m)
    prefactor = math.sqrt((2*l+1) / (4*math.pi))
    lastfactornumer = math.factorial(l+m) * math.factorial(l-m)
    lastfactordenom = math.factorial(l+1/2) * math.factorial(l-1/2)
    lastfactorsq = lastfactornumer / lastfactordenom
    factor = sign * prefactor * math.sqrt(lastfactorsq)
    return factor


def get_z(theta, phi):
    """Map spherical coordinates to the Riemann sphere."""
    return math.cos(phi)/math.tan(theta/2) + 1j*math.sin(phi)/math.tan(theta/2)


def EigSz(l, m, s, theta, phi):
    """Return the value of the eigenspinor |l,m,s> at theta, phi."""
    hsp = harmSpinorPrefactor(l, m)
    z = get_z(theta, phi)
    zb = np.conjugate(z)
    ql = (1+z*zb)**(-l)
    lm = int(l-1/2)
    lp = int(l+1/2)
    su = [scipy.special.binom(lm, r)*scipy.special.binom(lp,int(r+1/2-m))*z**r*(-1*zb)**(r+1/2-m) for r in range(0, lm+1)]
    sl = [scipy.special.binom(lp, r)*scipy.special.binom(lm,int(r-1/2-m))*z**r*(-1*zb)**(r-1/2-m) for r in range(0, lp+1)]
    spharpz = hsp*ql*sum(su)
    spharmz = hsp*ql*sum(sl)
    return 2**(-1/2)*np.array([s*spharpz, 1j*spharmz])


class memoize(object):
    """cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class memoizedSPHarmCoefficients:
    @memoize
    def threej(self, coeff):
        """Return the 3j-symbol (j1 j1 j2, m1 m2 m3)"""
        j1, m1, j2, m2, j3, m3 = coeff
        myargs = tuple(int(2*x) for x in (j1, j2, j3, m1, m2, m3))
        L1, L2, L3, M1, M2, M3 = myargs
        if M1 + M2 + M3 != 0:
            return 0
        elif abs(L1 - L2) > L3:
            return 0
        elif L1 + L2 < L3:
            return 0
        else:
            return py3nj.wigner3j(*myargs)

    def TripleProduct(self, j1, m1, s1, j2, m2, s2, j3, m3, s3):
        """Calculate int_{S^2} {}_{s_1} Y_{j1 m1} ... {}_{s_3} Y_{j3 m3}."""
        return math.sqrt((2*j1+1)*(2*j2+1)*(2*j3+1) / (4*math.pi)) * self.threej((j1, m1, j2, m2, j3, m3)) * self.threej((j1, -1*s1, j2, -1*s2, j3, -1*s3))

    def SPHarmComponents(self, l, m, l2, m2, s2, l3, m3, s3):
        """Return the triple integral {}_0 Y_{lm} {}_{s_2} Y_{l2m2}
        {}_{s3} Y_{l3 m3}."""
        s1 = 0
        j1 = l
        leftjms = (l2, m2, s2)
        rightjms = (l3, -1*m3, -1*s3)  # complex conjugation
        sign = (-1)**(m3+s3)
        return self.TripleProduct(j1, m, s1, *leftjms, *rightjms) * sign

    def SPHarmCoefficients(self, l, m,
                           l2, m2, diracsign_2,
                           l3, m3, diracsign_3):
        """Coefficients of Y_{lm} in the Dirac eigenspinor basis."""
        topleftjms = (l2, m2, 1/2)
        toprightjms = (l3, m3, 1/2)
        top = self.SPHarmComponents(l, m, *topleftjms, *toprightjms) * (diracsign_2 * diracsign_3/2)  # sign from top part (gb 9.57) and normalization
        bottomleftjms = (l2, m2, -1/2)
        bottomrightjms = (l3, m3, -1/2)
        bottom = self.SPHarmComponents(l, m, *bottomleftjms, *bottomrightjms) * (1/2)  # sign from bottom part (1j*-1j) and normalization
        return top + bottom


class TwoSphere:
    def __init__(self, spinorsize):
        self.spinorsize = spinorsize
        self.eigenspinors = geteigenspinors(spinorsize)
        self.P = self.getP()
        self.J = self.getJ()
        self.DiracS2 = self.getDiracS2()
        self.JPJ = self.getJPJ()
        self.G = self.getGamma()

    def getP(self):
        mat_b = matFromCoeffs(gen_bcoeff, self.eigenspinors)
        mat_a = matFromCoeffs(gen_acoeff, self.eigenspinors)
        ones = np.eye(self.spinorsize)
        P = [
            [(ones - mat_b)/2, mat_a/2],
            [mat_a.T/2, (ones + mat_b)/2]
            ]
        return np.block(P)

    def getGamma(self):
        G = matFromCoeffs(GammaCoeff, self.eigenspinors)
        return G

    def getJ(self):
        J = matFromCoeffs(JCoeff, self.eigenspinors)
        return J

    def getDiracS2(self):
        D = matFromCoeffs(diracS2Coeff, self.eigenspinors)
        return D

    def getJPJ(self):
        doubleJ = doubleMatrix(self.J)
        JPJ = np.linalg.multi_dot([doubleJ, self.P, doubleJ])
        return JPJ

    def leftRightActionCommutator(self, mat):
        """This is the commutator involved in the first-order condition."""
        mat = doubleMatrix(mat)
        ans = -comm(comm(mat, self.P), self.JPJ)
        return ans

    def leftRightActionCommutatorNorm(self, mat):
        lrac = self.leftRightActionCommutator(mat)
        ans = np.linalg.norm(lrac,'fro')
        return ans

    def oneSidedEquationDiff(self, mat):
        """This is the defect in the one-sided higher Heisenberg equation."""
        c = comm(doubleMatrix(mat), self.P)
        identity = np.identity(2*self.spinorsize)
        fullcomm = np.linalg.multi_dot([self.P-identity/2, c, c])
        diff = relTrace(fullcomm, self.spinorsize)-self.G
        return diff

    def scaledOneSidedEquationDiff(self, mat, eig=None,
                                   scalefunction=lambda x: (1+x**4)**(-1)):
        """Norm of the one-sided Heisenberg defect, with a regulator."""
        if eig is None:
            eig = np.linalg.eig(mat)
        evals, evects = eig
        u = np.matrix(evects)
        scalars = list(map(scalefunction, evals))
        factor = np.linalg.multi_dot([u, np.diag(scalars), u.H])
        return factor*self.oneSidedEquationDiff(mat)

    def chernAntiCommutator(self, mat):
        cc = self.oneSidedEquationDiff(mat)
        return cc*mat + mat*cc

    def SPHarm(self, l, m):
        """Return the matrix of the spherical harmonic Y_{lm} in the
        eigenspinor basis."""

        memSPHarCoef = memoizedSPHarmCoefficients()
        SPHarmCoefficients = memSPHarCoef.SPHarmCoefficients

        def mycoeffs(l0, m0, s0, l1, m1, s1):
            return SPHarmCoefficients(l, m, l0, m0, s0, l1, m1, s1)
        mat = matFromCoeffs(mycoeffs, self.eigenspinors)

        return mat
