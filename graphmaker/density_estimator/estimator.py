#!/usr/bin/python3

from ..smacof.smacof import WeightedDistanceGraph, SMACOF, run_embedding
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import math


def find_subgraph(distance_dict, center_node, radius):
    nodes = list_nodes(distance_dict)
    local_nodes = [node for node in nodes
                   if distance_dict[center_node, node] <= radius]
    local_nodes.sort(key=(lambda x: distance_dict[center_node, x]))
    return local_nodes


def list_nodes(distance_dict):
    return list(set([key[0] for key in distance_dict.keys()]))


def estimate_density(distance_dict, dimension, radius=None):
    if not radius:
        radius = estimate_radius(distance_dict)
    nodes = list_nodes(distance_dict)
    vols = {}
    rads = {}
    # minnpts = 4
    for node in nodes:
        local_nodes = find_subgraph(distance_dict, node, 2*radius)
        local_graph_mat = np.array([[distance_dict[node1, node2]
                                     for node1 in local_nodes]
                                    for node2 in local_nodes])
        wd = WeightedDistanceGraph(local_graph_mat,
                                   weight_function=lambda s, t, d: math.exp(-1*max([local_graph_mat[s, 0], local_graph_mat[t, 0]])))
        sm = SMACOF(wd, dimension=dimension)
        run_embedding(sm, tol=1e-8, num_steps=1000)
        embedding = sm.get_result_pointlist()
        embedding_center = embedding[0]
        uvects = np.identity(dimension)
        dirs = np.vstack((uvects, -1*uvects))
        embedding_bound = np.array([embedding_center + radius*direction
                                    for direction in dirs])
        embedding = np.vstack((embedding, embedding_bound))
        vor = Voronoi(embedding, qhull_options="Qz")
        vts = vor.vertices
        my_region = vor.regions[vor.point_region[0]]
        my_polytope = np.array([vts[k] for k in my_region])
        my_polytope_hull = ConvexHull(my_polytope)
        volume = my_polytope_hull.volume
        vols[node] = volume
        my_polytope_hull_vertices = [my_polytope_hull.points[k] for k in my_polytope_hull.vertices]
        rads[node] = max([np.linalg.norm(x - embedding_center)/2 for x in my_polytope_hull_vertices])

    return max(rads.values()), sum(vols.values())


def estimate_radius(distance_dict):
    nodes = list_nodes(distance_dict)
    return max([min([distance_dict[node, other_node]
                     for other_node in nodes
                     if other_node is not node])
                for node in nodes])


def inv_hball_vol(dimension, volume):
    prefactor = math.pi**(dimension/2) / math.gamma(dimension/2 + 1)
    return (volume/prefactor)**(1/dimension)


def estimate_density_radius(distance_dict, dimension, volume, radius=None):
    inrad, covervol = estimate_density(distance_dict, dimension, radius)
    vol_defect = abs(volume - covervol)
    outrad = inv_hball_vol(dimension, vol_defect)
    return inrad + outrad
