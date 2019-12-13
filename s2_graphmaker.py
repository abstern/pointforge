#!/usr/bin/python3

import argparse
import math
import multiprocessing
import copy
import time
import subprocess

from s2_init import construct_gmdata, get_Db
from graphmaker.graphmaker import (increment_graph, writeout,
                                   read_npz, as_distancearray,
                                   pprint)
from graphmaker.smacof.smacof import (SMACOF, run_embedding,
                                      WeightedDistanceGraph)
from graphmaker.smacof.smacof import writeout as smacof_write_bin


"""Construct and embed metric graphs associated to (PC(S^2)P,
PL^2(S^2,S), PD_{S^2}."""


def make_embedding(gmdata, outputbasename):
    """Run SMACOF to embed the graph in R^3."""
    darray = as_distancearray(gmdata)
    graph = WeightedDistanceGraph(darray,
                                  weight_function=lambda source, target, dist: math.exp(-dist))
    embedder = SMACOF(graph, dimension=3)

    run_embedding(embedder, num_steps=1000)
    smacof_write_bin(outputbasename, embedder)

    pprint("Writing output to {}.".format(
        outputbasename + " smacof.csv"))
    return


def parse_arguments():
    # Define command-line options
    description = "Calculate Connes distance for S2"
    parser = argparse.ArgumentParser(description=description)

    dim_help = "Dimension"
    input_help = "File with initial graph."
    output_help = "Basename for output files"
    parallel_help = "Number of parallel calculations"
    time_help = "Max calculation duration"
    mindistance_help = "Minimal distance between states"
    potential_help = "Strengths of repulsion between states"
    looping_help = "Run the graphmaker now!"
    hook_help = "Run $cmd upon adding data"

    def_threadnum = multiprocessing.cpu_count()
    parser.add_argument('--dim', metavar='dim', type=int,
                        default=12, help=dim_help)
    parser.add_argument('--input', metavar='input', type=str,
                        default=None, help=input_help)
    parser.add_argument('--output', metavar='output', type=str,
                        default=None, help=output_help)
    parser.add_argument('--threads', metavar='threads', type=int,
                        default=def_threadnum, help=parallel_help)
    parser.add_argument('--time', metavar='time', type=int,
                        default=60, help=time_help)
    parser.add_argument('--max-dispersion', metavar='max-dispersion',
                        type=float, default=0.3,
                        help=mindistance_help)
    parser.add_argument('--potential', metavar='electrostatic potential',
                        type=float, default=10,
                        help=potential_help)
    parser.add_argument('--loop', metavar="loop", type=int,
                        default=0, help=looping_help)
    parser.add_argument('--hook', metavar="hook", type=str,
                        default=None, help=hook_help)

    args = parser.parse_args()
    options = {}
    options['spinorsize'] = args.dim
    options['inputbasename'] = args.input
    options['outputbasename'] = args.output
    options['nthreads'] = args.threads
    options['duration'] = args.time
    options['max_dispersion'] = args.max_dispersion
    options['pot_coupling'] = args.potential
    options['loop'] = args.loop
    options['hook'] = args.hook
    return options


def run_hook(hook, outputbasename):
    """Run the program specified by hook with arg outputbasename."""
    if hook:
        subprocess.run([hook, outputbasename])


def loop_expand_big_graph(gmdataD, gmdataDb, outputbasename, duration, hook):
    start_time = time.time()

    if outputbasename:
        outputD = outputbasename
        outputDb = outputbasename + " Db"

    def elapsed_time():
        return int(time.time() - start_time)

    while elapsed_time() < duration and len(gmdataD.states) < gmdataD.maxstates:
        pprint("Have {} states. Stepping once. {} seconds left.".format(
            len(gmdataD.states),
            duration - elapsed_time()))
        increment_graph(gmdataD, steps=1)
        if outputbasename:
            writeout(gmdataD, outputD)
        pprint("Working on Db...")
        increment_graph(gmdataDb, steps=0)
        if outputbasename:
            writeout(gmdataDb, outputDb)

        run_hook(hook, outputbasename)

    if outputbasename:
        make_embedding(gmdataD, outputD)
        make_embedding(gmdataDb, outputDb)
        run_hook(hook, outputbasename)

    return


if __name__ == "__main__":
    options = parse_arguments()
    pprint("Constructing gmdata...")
    gmdata = construct_gmdata(**options)

    Db = get_Db(gmdata.D)
    gmdata_Db = copy.copy(gmdata)
    # separate distance, Dirac operator
    gmdata_Db.distances = {}
    gmdata_Db.D = Db
    pprint("Done initializing.")

    if options['inputbasename'] is not None:
        pprint("Reading from {}...".format(options['inputbasename']))
        try:
            states, distances = read_npz(options['inputbasename'])
            statesDb, distancesDb = read_npz(options['inputbasename'] + " Db")

            gmdata.states = states
            gmdata_Db.states = gmdata.states

            gmdata.distances = distances
            gmdata_Db.distances = distancesDb

        except FileNotFoundError:
            print("Input file does not exist, starting from scratch.")

    outputbasename = options['outputbasename']
    hook = options['hook']

    def loop_once(duration=options['duration']):
        loop_expand_big_graph(gmdata, gmdata_Db,
                              outputbasename, duration, hook)

    if options['loop']:
        loop_once(options['duration'])
