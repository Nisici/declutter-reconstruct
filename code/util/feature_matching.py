import networkx as nx
from networkx.algorithms.matching import min_weight_matching
from networkx.algorithms import bipartite
from object.Segment import segments_distance
from object.Segment import radiant_inclination
from object.Segment import p_to_p_dist
from object.Segment import Segment
import numpy as np
from random import randint
import cv2
from matplotlib import pyplot as plt
"""
Given two lists of ExtendedSegment find the minimum weight matching between them, where the weight
is the distance between a line from line1 and a line from line2. If two lines have an angular distance that is
too big the weight between them is very large because there shouldn't be a matching between them.
Returns: list of matchings [(l_i, l_j), ...]
doesn't work
"""
from scipy.optimize import linear_sum_assignment

#returns the lines from the list that follow the input direction
def lines_of_direction(lines, dir):
    lines_of_dir = []
    tol = 0.1
    print("DIR: {}".format(dir))
    for l1 in lines:
        angle = abs(radiant_inclination(l1.x1, l1.y1, l1.x2, l1.y2))
        if abs(angle - dir) < tol:
            lines_of_dir.append(l1)
    return lines_of_dir
def lines_matching(lines1, lines2, dirs1, dirs2):
    #M = matrix NxN of features
    # M[i,j] = distance between i and j, if i != j
    # M[i,j] = inf if i=j
    #weighted_edges = []
    def make_weight_matrix(lines1, lines2):
        weighted_edges = np.zeros((len(lines1), len(lines2)))
        for i in range(len(lines1)):
            for j in range(len(lines2)):
                l1 = lines1[i]
                l2 = lines2[j]
                angle1 = radiant_inclination(l1.x1, l1.y1, l1.x2, l1.y2)
                angle2 = radiant_inclination(l2.x1, l2.y1, l2.x2, l2.y2)
                # if they have different directions they can't be matched.
                if abs(angle1 - angle2) > np.pi / 4:
                    weight = 5000
                else:
                    weight = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
                # w_e = (l1, l2, weight)
                # weighted_edges.append(w_e)
                weighted_edges[i, j] = weight
        return weighted_edges
    horizontal_lines1 = lines_of_direction(lines1, dirs1[0])
    horizontal_lines2 = lines_of_direction(lines2, dirs2[0])
    vertical_lines1 = lines_of_direction(lines1, dirs1[1])
    vertical_lines2 = lines_of_direction(lines2, dirs2[1])
    weighted_edges_horiz = make_weight_matrix(horizontal_lines1, horizontal_lines2)
    weighted_edges_vert = make_weight_matrix(vertical_lines1, vertical_lines2)
    row_ind_horiz, col_ind_horiz = linear_sum_assignment(weighted_edges_horiz)
    row_ind_vert, col_ind_vert = linear_sum_assignment(weighted_edges_vert)
    avg_dist = (weighted_edges_horiz[row_ind_horiz, col_ind_horiz].sum() +
               weighted_edges_vert[row_ind_vert, col_ind_vert].sum()) / min(len(lines1), len(lines2))
    return avg_dist

# dirs1: (m1, m2)
# dirs2: (n1, n2)
# dist = (|m1 - n1|, |m2 - n2|)
def distance_between_directions(dirs1, dirs2):
    return 0
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
    """
    for match in matchings:
        l1 = match[0]
        l2 = match[1]
        distance = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
        distances.append(distance)
    """
    for l1, l2 in matchings.items():
        distance = segments_distance(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
        distances.append(distance)
    print("Distances len: {}, distances: {}".format(len(distances), distances))
    return np.mean(distances)

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
