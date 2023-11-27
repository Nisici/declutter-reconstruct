import networkx as nx
from networkx.algorithms.matching import min_weight_matching
from networkx.algorithms import bipartite
from object.Segment import segments_distance
from object.Segment import radiant_inclination
from object.Segment import p_to_p_dist
from object.Segment import Segment
import numpy as np

"""
Given two lists of ExtendedSegment find the minimum weight matching between them, where the weight
is the distance between a line from line1 and a line from line2. If two lines have an angular distance that is
too big the weight between them is very large because there shouldn't be a matching between them.
Returns: list of matchings [(l_i, l_j), ...]
"""
def lines_matching(lines1, lines2):
    #M = matrix NxN of features
    # M[i,j] = distance between i and j, if i != j
    # M[i,j] = inf if i=j
    G = nx.Graph()
    G.add_nodes_from(lines1, bipartite=0)
    G.add_nodes_from(lines2, bipartite=1)
    const = 0.5
    weighted_edges = []
    for l1 in lines1:
        for l2 in lines2:
            angle1 = radiant_inclination(l1.x1, l1.y1, l1.x2, l1.y2)
            angle2 = radiant_inclination(l2.x1, l2.y1, l2.x2, l2.y2)
            angle1 = angle1%np.pi
            angle2 = angle2%np.pi
            if l1 == l2:
                weight = 5000
            #if they have different directions they can't be matched.
            elif abs(angle1 - angle2) > np.pi/2 - const :
                weight = 5000
            else:
                weight = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
            w_e = (l1, l2, weight)
            weighted_edges.append(w_e)
    G.add_weighted_edges_from(weighted_edges)
    matchings = min_weight_matching(G)
    return matchings

"""
Matching between contours.
"""
def contour_matching(contour1, contour2):
    G = nx.Graph()
    contour1 = contour1.copy()
    contour2 = contour2.copy()
    lines1 = []
    lines2 = []
    for i in range(len(contour1)-1):
        line = Segment(contour1[i][0], contour1[i][1], contour1[i+1][0], contour1[i+1][1])
        lines1.append(line)
    for i in range(len(contour2)-1):
        line = Segment(contour2[i][0], contour2[i][1], contour2[i+1][0], contour2[i+1][1])
        lines2.append(line)
    G.add_nodes_from(lines1, bipartite=0)
    G.add_nodes_from(lines2, bipartite=1)
    weighted_edges = []
    for l1 in lines1:
        for l2 in lines2:
            weight = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
            w_e = (l1, l2, weight)
            weighted_edges.append(w_e)
    G.add_weighted_edges_from(weighted_edges)
    matchings = min_weight_matching(G)
    return matchings

def average_distance_between_points(matchings):
    distances = []
    for match in matchings:
        p1 = match[0]
        p2 = match[1]
        distance = p_to_p_dist(p1, p2)
        distances.append(distance)
    return np.mean(distances)

def average_distance_between_lines(matchings):
    distances = []
    for match in matchings:
        l1 = match[0]
        l2 = match[1]
        distance = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
        distances.append(distance)
    return np.mean(distances)
