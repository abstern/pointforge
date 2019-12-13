#!/usr/bin/python3

import numpy as np
import math
import itertools


class WeightedDistanceGraph:
    def __init__(self, matrix,
                 weight_function=lambda source, target, length: 1):
        assert matrix.shape[0] == matrix.shape[1]
        self.weight_function = weight_function
        self.size = matrix.shape[0]
        self.nodes = range(self.size)
        self.matrix = matrix
        self.edges = self.find_edges()

    def find_edges(self):
        edges = {}
        pairs = np.isfinite(self.matrix)
        total_weight = 0
        for pair in itertools.combinations(self.nodes, r=2):
            n, m = pair
            if pairs[n, m]:
                dist = self.matrix[n, m]
                edges[n, m] = dist, self.weight_function(n, m, dist)
                total_weight += self.weight_function(n, m, dist)
        edges = {key: (value0, 1/total_weight * value1)
                 for key, (value0, value1) in edges.items()}
        return edges

    def walk_nodes(self):
        return self.nodes

    def walk_edges(self):
        return self.edges.items()


class SMACOF:
    def __init__(self, graph, dimension=3):
        self.dimension = dimension
        self.graph = graph
        num_nodes = len(self.graph.walk_nodes())
        self.conf_shape = (num_nodes, self.dimension)
        self.b_shape = (num_nodes, num_nodes)
        self.z = self.initial_z()
        self.x = self.z
        self.vplus = find_mp_hessian(self.graph)



    def initial_z(self):
        z = np.random.rand(*self.conf_shape)
        return z


    def position(self, node, configuration):
        return configuration[node,...]


    def distance(self, node1, node2, configuration):
        p1 = self.position(node1, configuration)
        p2 = self.position(node2, configuration)
        return np.linalg.norm(p1 - p2)


    def guttman(self, z):
        b = self.find_b(z)
        return np.linalg.multi_dot([self.vplus, b, z])


    def find_b(self, z):
        b = np.zeros(self.b_shape)
        for (n, m), (dist, weight) in self.graph.walk_edges():
           dz = self.distance(n, m, z)
           if not math.isclose(dz, 0):
               entry = - weight * dist / dz
               b[n, m] = entry
               b[m, n] = entry
           else:
               b[n, m] = 0
               b[m, n] = 0
        for node in self.graph.walk_nodes():
            n = node
            b[n, n] = - np.sum(b, axis = 1)[n]
        return b


    def stress(self):
        disterrs = [weight*(self.distance(n, m, self.x)-dist)**2 for (n, m), (dist, weight) in self.graph.walk_edges()]
        return sum(disterrs)


    def step(self):
        self.x = self.guttman(self.z)
        self.z = self.x

    def get_result_pointlist(self):
        return self.x



class SphereShellSMACOF(SMACOF):
    def __init__(self, graph):
        SMACOF.__init__(self, graph, dimension = 2)


    def great_circle_distance(self, p1, p2):
        return math.acos(math.sin(p1[0])*math.sin(p2[0]) + math.cos(p1[0])*math.cos(p2[0])*math.cos(p1[1]-p2[1]))

    def distance(self, node1, node2, configuration):
        p1 = self.position(node1, configuration)
        p2 = self.position(node2, configuration)
        return self.great_circle_distance(p1, p2)

        return np.linalg.norm(p1 - p2)

    def stress(self):
        e = 0
        for (n, m), (dist, weight) in self.graph.walk_edges():
            d = self.distance(n, m, self.x)
            e += weight*(dist - d)**2
        return e


def find_hessian(graph):
    v = np.zeros((graph.size, graph.size))
    for (n, m), (dist, weight) in graph.walk_edges():
        a = np.zeros((graph.size, graph.size))
        a[n, n] = 1
        a[m, m] = 1
        a[n, m] = -1
        a[m, n] = -1
        w = weight
        v += w*a
    return v


def find_mp_hessian(graph):
    hessian = find_hessian(graph)
    return np.linalg.pinv(hessian)


def chunker(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def run_embedding(embedding, num_steps=100, tol=1e-12):
    steplist = range(num_steps)
    stepset = chunker(steplist, 10)
    for chunk in stepset:
        prev_stress = embedding.stress()
        for step in chunk:
            embedding.step()
        # print("Stress: {}".format(embedding.stress()))
        if abs(embedding.stress() - prev_stress) < tol:
            # print("SMACOF finished: tolerance reached.")
            break
    else:
        print("SMACOF finished: maximum number of steps reached!")


def cutoff_entries(distancearray, epsilon):
    newarray = np.copy(distancearray)
    newarray[newarray > epsilon] = None
    return newarray


def writeout(outputbasename, embedding):
    coordinates = embedding.x
    np.savetxt(outputbasename + " smacof.csv", coordinates, delimiter = ',')
    return


def my_weight(length):
    return math.exp(-length)


def read_bin_states(outputbasename):
    distancearray = np.fromfile(outputbasename, dtype = np.float_)
    l = distancearray.shape[0]
    d = int(math.sqrt(l))
    distancearray = distancearray.reshape((d, d))
    vectarray = np.fromfile(outputbasename + " states", dtype = np.complex_)
    disparray = np.fromfile(outputbasename + " dispersions", dtype = np.complex_)
    numstates = vectarray.shape[0]

    return distancearray
