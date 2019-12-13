#!/usr/bin/python

import numpy as np
import scipy.optimize
import scipy.linalg
import functools
import math


"""Generate localized states of an operator system spectral triple
(PAP, PH, PD)"""


def complexform(vect):
    # emulate complex vector space (of half the dimension) for numpy.optimize
    myl = int(vect.shape[0]/2)
    return vect[:myl] + 1j*vect[myl:]


def apply_state(vect, mat):
    # vect, mat -> <v|mat|v>
    return np.inner(vect, np.dot(mat, vect.conjugate()))


def stateform(vect):
    # vect -> (mat -> <v|mat|v>)
    v = complexform(vect)
    return functools.partial(apply_state, v)


class State:
    """Given v, store the state <v, - v>. Can be hashed."""

    def __init__(self, vect, potential_at_creation=None, dispersion=None, **kwargs):
        self.function = stateform(vect)
        self.vector = complexform(vect)
        self.potential_at_creation = potential_at_creation
        self.dispersion = dispersion

    def __key(self):
        return tuple(self.vector.tolist())

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class DispersionParameters:
    def __init__(self, coordinates, sq_coordinates, **kwargs):
        self.coordinates = coordinates
        self.sq_coordinates = sq_coordinates


def dispersion(params, state):
        "Calculate the dispersion, i.e. the variance of position."""
        esqs = (state(sq_coordinate) for sq_coordinate in
                params.sq_coordinates)
        sqes = (state(coordinate)**2 for coordinate in
                params.coordinates)
        d = abs(sum(esqs) - sum(sqes))
        return d


class PotentialParameters(DispersionParameters):
    def __init__(self, coordinates, sq_coordinates, states=[],
                 min_dispersion=0, pot_coupling=3, **kwargs):
        DispersionParameters.__init__(self, coordinates, sq_coordinates)
        self.states = states
        self.min_dispersion = min_dispersion
        self.pot_coupling = pot_coupling


def potential(params, vect):
    state = stateform(normalize(vect))
    # modeled as a pair of opposite charges of separation disp:
    # electrostatic attraction
    attraction = -1/(dispersion(params, state) + params.min_dispersion)
    repulsion = (1/coord_distance(params.coordinates, state, alt_state)
                 for alt_state in params.states)
    return attraction + params.pot_coupling*sum(repulsion)


def normalize(vect):
    """Normalize non-zero vector."""
    norm = np.linalg.norm(vect)
    return vect / norm


def coord_distance(coordinates, state1, state2):
    """Return euclidean distance between states."""
    distsqs = ((state1(coordinate) - state2(coordinate)).real**2 for
               coordinate in coordinates)
    return math.sqrt(sum(distsqs))


class StateFinderParameters(PotentialParameters):
    def __init__(self, **kwargs):
        PotentialParameters.__init__(self, **kwargs)
        self.size = 2*self.coordinates[0].shape[0]
        constraint_lb = 0.8
        constraint_ub = 1.2
        self.constraint = scipy.optimize.NonlinearConstraint(
            fun=self.constraint_fun,
            lb=constraint_lb, ub=constraint_ub)
        self.reset_x0()

    def reset_x0(self):
        init_vect = np.random.rand(self.size)
        self.init_vect = init_vect / np.linalg.norm(init_vect)

    def constraint_fun(self, vect):
        return np.linalg.norm(vect)


def find_new_state(params):
    """Find a new state, as local minimum of the potential."""
    opts = {'maxiter': 3000}
    mypotential = functools.partial(potential, params)
    bounds = [(-1, 1) for x in range(params.size)]
    result = scipy.optimize.minimize(fun=mypotential,
                                     x0=params.init_vect,
                                     constraints=params.constraint,
                                     options=opts, bounds=bounds)
    if not result.success:
        print(result.message)
    vect = normalize(result.x)
    state = State(vect, potential_at_creation=mypotential(vect))
    state.dispersion = dispersion(params, state)
    return state
