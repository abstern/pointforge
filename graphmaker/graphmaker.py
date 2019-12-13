#!/usr/bin/python3

import numpy as np
from .connes_distance import DistanceParameters, find_distance
from .localized_states import StateFinderParameters, State, find_new_state
from .spectral_invariants.invariants import dimension, volume
from .density_estimator.estimator import estimate_density_radius
import math
import itertools
from multiprocessing import Pool
import functools
import time
import operator

"""Friendly helper library around connes_distance and localized_states.

increment_graph() creates a distance graph, write_bin/npz() and read_bin/npz()
write/read a graph from disk."""


def round_to_zero(myarray, tolerance):
    """Round array to zero within tolerance. Is useful for sparsity."""

    smallreals = np.abs(myarray.real < tolerance)
    smallimags = np.abs(myarray.imag < tolerance)
    result = myarray
    result.real[smallreals] = 0
    result.imag[smallimags] = 0
    return result


class GraphmakerData(DistanceParameters,
                     StateFinderParameters):
    """Contains both the distance graph and the parameters required to
    grow it."""

    def __init__(self, distances={}, nthreads=1, **kwargs):
        DistanceParameters.__init__(self, **kwargs)
        StateFinderParameters.__init__(self, **kwargs)
        self.distances = distances
        self.nthreads = nthreads
        self.evals = np.linalg.eigvals(self.D)
        self.maxstates = estimate_state_num_bound(self.evals)


def append_state_inplace(gmdata):
    state = find_new_state(gmdata)
    gmdata.states.append(state)
    gmdata.reset_x0()
    fb_string = "State found, dispersion {:5.3f}, potential {:6.3f}."
    pprint(fb_string.format(state.dispersion,
                            state.potential_at_creation))


def star_find_distance(gmdata, combination):
    return find_distance(gmdata, *combination)


def find_distances_inplace(gmdata, combinations):
    """Calculate distances from list of pairs."""
    fb_string = """getdistances got {} combinations, calculating..."""
    pprint(fb_string.format(len(combinations)))

    distfinder = functools.partial(star_find_distance, gmdata)

    with Pool(gmdata.nthreads) as p:
        result = p.imap(distfinder, combinations)
        for pair, distance in zip(combinations, result):
            gmdata.distances.update({pair: distance})
            gmdata.distances.update({tuple(reversed(pair)): distance})
        pprint("...done.")


def finish_distance_graph_inplace(gmdata):
    """Calculate missing distances, append to gmdata.distances."""
    states, dists = gmdata.states, gmdata.distances
    trivial_distances = {(state, state): 0 for state in states}
    dists.update(trivial_distances)
    combinations = itertools.combinations(states, r=2)
    existing_combinations = [pair for pair in dists.keys() if dists[pair]]
    missing_combinations = [pair for pair in combinations if pair not
                            in existing_combinations]
    find_distances_inplace(gmdata, missing_combinations)


def pprint(string):
    prefix = time.strftime("%x %X") + ": "
    print(prefix + string, flush=True)


def increment_graph(gmdata, steps=1):
    """Increment the graph in gmdata."""

    for step in range(steps):
        pprint("Adding state...")
        append_state_inplace(gmdata)
    pprint("Updating distances...")
    finish_distance_graph_inplace(gmdata)
    pprint("Succeeded, appending: we have {} states.".format(
        len(gmdata.states)))


def as_distancearray(gmdata):
    """Return distances as array, in order distancearray[n, m] =
    distances[gmdata.states[n], gmdata.states[m]]."""

    data = as_data_arrays(gmdata)
    return data['distances']


def as_data_arrays(gmdata):
    """Export states, distances to numpy arrays, such that
    data['distances'][n, m] = distances[data['states'][n],
    data['states'][m]]."""

    states = gmdata.states
    distances = gmdata.distances

    dim = gmdata.D.shape[0]

    vectshape = (len(states), dim)
    vectarray = np.full(vectshape, np.nan, dtype=np.complex_)
    potarray = np.full(len(states), np.nan, dtype=np.float_)
    disparray = np.full(len(states), np.nan, dtype=np.float_)

    distshape = (len(states), len(states))
    distancearray = np.full(distshape, np.nan, dtype=np.float_)

    for n, state1 in enumerate(states):
        vectarray[n] = state1.vector
        disparray[n] = state1.dispersion
        potarray[n] = state1.potential_at_creation
        for m, state2 in enumerate(states):
            if (state1, state2) in distances.keys():
                distancearray[n, m] = distances[state1, state2]
    data = {}
    if states:
        data['vectors'] = vectarray
        data['dispersions'] = disparray
        data['potentials'] = potarray
    if distances:
        data['distances'] = distancearray

    return data


def writeout(gmdata, basename):
    """Write to .npz file for resuming and to .dat file for easy
    Mathematica read-in."""

    write_npz(gmdata, basename)
    write_bin(gmdata, basename)


def write_bin(gmdata, basename):
    data = as_data_arrays(gmdata)
    varray = data['vectors'].T
    varray.tofile(basename + "_states.dat")
    data['distances'].tofile(basename + "_distances.dat")


def write_npz(gmdata, basename):
    """Write distances, _states, _dispersions, _potentials to single
    .npz file."""
    data = as_data_arrays(gmdata)
    np.savez_compressed(basename + ".npz", **data)


def from_data_arrays(data):
    """Transform data dictionary into states, distances objects."""
    states, distances = [], {}
    try:
        vectarray = data['vectors']
        disparray = data['dispersions']
        potarray = data['potentials']
        distancearray = data['distances']

        statedict = {}
        distances = {}
        iterator = enumerate(zip(vectarray, potarray, disparray))
        for n, (vect, pot, disp) in iterator:
            splitvect = np.block([vect.real, vect.imag])
            newstate = State(splitvect,
                             potential_at_creation=pot,
                             dispersion=disp)
            statedict[n] = newstate
        states = list(statedict.values())
        for n, state1 in statedict.items():
            for m, state2 in statedict.items():
                try:
                    distances[state1, state2] = distancearray[n, m]
                except IndexError:
                    print("Distance {}, {}: not yet calculated.".format(n, m))
    except KeyError:
        print("Missing part of the data.")

    return states, distances


def read_npz(basename):
    """Read from .npz output."""
    data = np.load(basename + ".npz")
    states, distances = from_data_arrays(data)
    return states, distances


def read_bin(basename):
    """Read from (older) numpy binary output."""
    data = {}
    cpkeys = {'vectors': "_states"}
    flkeys = {'dispersions': '_dispersions',
              'potentials': '_potentials', 'distances': ''}
    for key, val in cpkeys.items():
        data[key] = np.fromfile(basename + val, dtype=np.complex_)
    for key, val in flkeys.items():
        data[key] = np.fromfile(basename + val, dtype=np.float_)

    nstates = len(data['potentials'])
    ndistances = int(math.sqrt(data['distances'].size))
    data['distances'] = data['distances'].reshape((ndistances, ndistances))
    data['vectors'] = data['vectors'].reshape((-1, nstates)).T

    states, distances = from_data_arrays(data)
    return states, distances


def estimate_state_num_bound(evals):
    dim = estimate_dimension(evals)
    vol = volume(evals, dim=dim)
    ebv = estimate_euclidean_ball_volume(dim)

    cutoff = max(evals)
    euc_disp = estimate_euclidean_dispersion(cutoff, dim)
    
    return vol / (ebv * euc_disp**(dim/2))


def estimate_dimension(evals):
    return int(dimension(evals))


def estimate_euclidean_dispersion(cutoff, dim):
    cov_diag = math.log(cutoff)/cutoff**2
    cov_trace = dim*cov_diag
    return cov_trace


def estimate_euclidean_ball_volume(dim):
    if dim%2==0:
        k = dim/2
        denom = math.factorial(k)
        numer = math.pi**k
    else:
        k = dim/2+1/2
        denom = 2*math.factorial(k) * 4* math.pi**k
        numer = math.factorial(2*k+1)
        
    return numer / denom
