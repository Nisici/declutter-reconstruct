import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import object.Segment as Segment
import util.disegna as dsg
from random import randint
from object.Segment import radiant_inclination, segments_distance
from util.feature_matching import lines_of_direction
from skimage.metrics import structural_similarity
from sklearn.metrics import jaccard_score
import os
from util.disegna import draw_contour
from util.layout import external_contour

#calculate EDM between two distributions in form of dictionaries dict[num_of_pix_walls] = distance
# distance matrix built using L2 distance, emd normalized dividing by total number of pixels
def EMD_walls_distances(dict1, dict2):
    print("Num pixels dict1: {}".format(sum(dict1.values())))
    print("Num pixels dict2: {}".format(sum(dict2.values())))
    num_of_pixels = sum(dict1.values())
    dist_matrix = make_distance_matrix(dict1, dict2, metric='euclidean')
    distrib_walls_lines = np.reshape(list(dict1.values()), (-1, 1))
    distrib_walls_corrected_lines = np.reshape(list(dict2.values()), (-1, 1))
    distrib_walls_lines = distrib_walls_lines.flatten().astype('float64')
    distrib_walls_corrected_lines = distrib_walls_corrected_lines.flatten().astype('float64')
    emd, flow = pyemd.emd_with_flow(distrib_walls_lines, distrib_walls_corrected_lines, dist_matrix)
    emd = (emd**0.5) / num_of_pixels
    return emd

#calculate EDM between two distributions in form of dictionaries dict[num_of_walls] = direction
# distance matrix built using L1 angle distance, emd normalized dividing by total number of walls
def EMD_walls_directions(dict1, dict2):
    print("Num pixels dict1 angular: {}".format(sum(dict1.values())))
    print("Num pixels dict2 angular: {}".format(sum(dict2.values())))
    num_of_walls = sum(dict1.values())
    dist_matrix = make_distance_matrix(dict1, dict2, 'angular')
    distrib_walls_angular = np.reshape(list(dict1.values()), (-1, 1))
    distrib_walls_angular_corrected = np.reshape(list(dict2.values()), (-1, 1))
    distrib_walls_angular = distrib_walls_angular.flatten().astype('float64')
    distrib_walls_angular_corrected = distrib_walls_angular_corrected.flatten().astype('float64')
    emd, flow = pyemd.emd_with_flow(distrib_walls_angular, distrib_walls_angular_corrected, dist_matrix)
    emd = emd / num_of_walls
    return emd

def distance_point_line(line, point_x, point_y):
    return np.abs(
        (line.y1 - line.y2) * point_x - (line.x1 - line.x2) * point_y + line.x1 * line.y2 - line.y1 * line.x2) / np.sqrt(
        (line.y1 - line.y2) ** 2 + (line.x1 - line.x2) ** 2
    )

# union distribution of walls and lines distances (not pixels but walls)
def walls_distance_distribution(walls, lines):
    count = {0: 0}
    for line in lines:
        spatial_cluster = [w for w in walls if w.spatial_cluster == line.spatial_cluster]
        tol = 0.1
        for w in spatial_cluster:
            distance = segments_distance(w.x1, w.y1, w.x2, w.y2, line.x1, line.y1, line.x2, line.y2)
            flag = False
            for d in count.keys():
                if abs(distance - d) < tol:
                    count[d] += 1
                    flag = True
            if not flag:
                count[round(distance, 1)] = 1
    return count

# union distribution of walls' pixels and lines distances
def pixels_walls_distance_distribution(walls, lines, original_binary_map):
    count = {0: 0}
    for line in lines:
        spatial_cluster = [w for w in walls if w.spatial_cluster == line.spatial_cluster]
        pixels = pixel_to_wall(original_binary_map, spatial_cluster)
        tol = 0.1
        for (x, y) in pixels:
            distance = distance_point_line(line, x, y)
            flag = False
            for d in count.keys():
                if abs(distance - d) < tol:
                    count[d] += 1
                    flag = True
                    break
            if not flag:
                #count[round(distance, 3)] = 1
                count[distance] = 1
    return count

# union distribution of walls' direction, histogram of directions (x) and number of walls who have that direction (y)
# normalized in range (-pi/2, pi/2]
# in the future you should do the distribution based on pixels (using original_binary_map) of the walls and not the walls
def walls_directions_distribution(walls, original_binary_map):
    pi = 3.14
    count = {-pi/2: 0, 0: 0, pi/2: 0}
    tol = 0.1
    for w in walls:
        angle = radiant_inclination(w.x1, w.y1, w.x2, w.y2)
        if angle > pi/2 and angle <= 3/2*pi :
            angle -= pi
        elif angle <= -pi/2 and angle >= -3/2*pi:
            angle += pi
        flag = False
        for a in count.keys():
            if abs(angle - a) < tol:
                count[a] += 1
                flag = True
                break
        if not flag:
            count[angle] = 1
    return count
"""
walls: set of ExtendedSegment
lines: set of ExtendedSegment
dirs: main directions of lines
compute the average distance between the walls and the associated lines.
"""
def avg_distance_walls_lines(walls, lines, original_binary_map, dirs):
    distances = {}
    walls_without_out = remove_walls_outliers(walls, dirs)
    pixels_to_walls = pixel_to_wall(original_binary_map, walls_without_out)
    for (x, y), wall in pixels_to_walls.items():
        lines_spatial_clust = [l for l in lines if l.spatial_cluster == wall.spatial_cluster]
        line = lines_spatial_clust[0]
        distance = distance_point_line(line, x, y)
        distances[wall] = distance
    return np.mean(list(distances.values()))

# returns: dictionary[pixel] = wall
def pixel_to_wall(original_binary_map, walls):
    mask = np.zeros_like(original_binary_map)
    # wall color in the binary map
    wall_white = 150
    # Create a dictionary to associate each pixel with the wall that masked it
    pixel_to_wall = {}
    # Iterate through the detected lines and draw them on the mask while associating pixels
    for wall in walls:
        # Draw line segment on mask
        cv2.line(mask, (int(wall.x1), int(wall.y1)), (int(wall.x2), int(wall.y2)), 255, 2)
        # Iterate through the pixels covered by this line segment
        for y in range(min(int(wall.y1), int(wall.y2)), max(int(wall.y1), int(wall.y2)) + 1):
            for x in range(min(int(wall.x1), int(wall.x2)), max(int(wall.x1), int(wall.x2)) + 1):
                # white perchè nelle mappe create automaticamente quello è il valore che corrisponde al bianco nelle mappe binary.
                if (mask[y, x] == 255) and (original_binary_map[y, x] >= wall_white):
                    pixel_to_wall[(x, y)] = wall
    return pixel_to_wall

#remove walls that have an inclination that is too much different from the main directions,
#this is decided by a threshold
def remove_walls_outliers(walls, dirs):
    tol = 0.1
    walls_without_outl = []
    p0 = dirs[0] + np.pi / 2
    p1 = dirs[1] + np.pi / 2
    p0 = p0 % (2 * np.pi)
    p1 = p1 % (2 * np.pi)
    p0 = np.pi - p0
    p1 = np.pi - p1
    for w in walls:
        angle = radiant_inclination(w.x1, w.y1, w.x2, w.y2)
        if abs(angle - dirs[0]) <= tol or abs(angle - dirs[1]) <= tol\
                or abs(angle - p0) <= tol or abs(angle - p1) <= tol:
            walls_without_outl.append(w)
    return walls_without_outl

def distance_heat_map(lines, walls, filepath, original_map, original_binary_map, main_dirs):
    def distance_point_line(line, point_x, point_y):
        return np.abs(
            (line.y1 - line.y2) * point_x - (
                        line.x1 - line.x2) * point_y + line.x1 * line.y2 - line.y1 * line.x2) / np.sqrt(
            (line.y1 - line.y2) ** 2 + (line.x1 - line.x2) ** 2
        )

    max_distance = 0
    min_distance = 0  # arbitrary
    heatmapOriginal = original_map.copy()
    if len(heatmapOriginal.shape) == 2:
        heatmapOriginal = cv2.cvtColor(heatmapOriginal, cv2.COLOR_GRAY2RGB)
    walls = remove_walls_outliers(walls, main_dirs)
    pix_to_wall = pixel_to_wall(original_binary_map, walls)
    for (x, y), wall in pix_to_wall.items():
        # Get the associated line
        lines_spatial_clust = [l for l in lines if l.spatial_cluster == wall.spatial_cluster]
        if len(lines_spatial_clust) > 1:
            print("Number of lines associated with this wall: {}".format(len(lines_spatial_clust)))
        line = lines_spatial_clust[0]
        distance = distance_point_line(line, x, y)
        # Calculate the distance of the pixel from the line using the point-line distance formula
        if distance > max_distance:
            max_distance = distance
    colormap = plt.get_cmap('jet')
    colormap = colormap.reversed()
    heatmap = np.zeros_like(original_map, shape=(original_map.shape[0], original_map.shape[1], 3))
    for (x, y), wall in pix_to_wall.items():
        lines_spatial_clust = [l for l in lines if l.spatial_cluster == wall.spatial_cluster]
        line = lines_spatial_clust[0]
        distance = distance_point_line(line, x, y)
        if max_distance == 0 and min_distance == 0:
            normalized = 0
        else:
            normalized = (distance - min_distance) / (max_distance - min_distance)
        # Map the distance to a color in the colormap
        # normalized = distance/max_distance
        color = colormap(normalized)
        # Set the color in the heatmap
        heatmapOriginal[y, x] = (color[0] * 255, color[1] * 255, color[2] * 255)
        heatmap[y, x] = (color[0] * 255, color[1] * 255, color[2] * 255)
    plt.imsave(os.path.join(filepath, 'heatmap.png'), heatmap)
    plt.imsave(os.path.join(filepath, 'heatmap_on_original.png'), heatmapOriginal)
    return heatmap


"""
#distribution of distances of walls given a line
def walls_distance_distribution(walls, line, original_binary_map):
    count = {0:0}
    spatial_cluster = [w for w in walls if w.spatial_cluster == line.spatial_cluster]
    pixels = pixel_to_wall(original_binary_map, spatial_cluster)
    tol = 0.1
    for (x, y) in pixels.keys():
        distance = distance_point_line(line, x, y)
        distance = normalize_distance(distance)
        flag = False
        for d in count.keys():
            if abs(distance - d) < tol:
                count[d] += 1
                flag = True
        if not flag:
            count[round(distance, 1)] = 1
    return count
"""

"""
OLD ONE, use distance_heat_map
Calculate the distance heat map between walls and walls projections and draw it.
walls_projections: array of walls projections calculated in the Segment.set_weights() file.
pixel_to_wall:     a dictionary where each pixel that belongs to a wall gets associated with its wall.
                   {(x_1, y_1): wall), (x_2, y_2: wall). . .}
the heat map is constructed finding the distance between each pixel and its wall_projection, colors are given
using the jet colormap (dark red means near, dark blue means distant), normalizing it based on the maximum distance
that has been computed using these pixels and walls.

"""
def distance_heat_map_wall_proj(walls_projections, pixel_to_wall, filepath, original_map, name):
    def distance_point_line(line, point_x, point_y):
        return np.abs(
            (line.y1 - line.y2) * point_x - (line.x1 - line.x2) * point_y + line.x1 * line.y2 - line.y1 * line.x2) / np.sqrt(
            (line.y1 - line.y2) ** 2 + (line.x1 - line.x2) ** 2
        )
    max_distance = 0
    min_distance = 0 #arbitrary
    heatmapOriginal = original_map.copy()
    if len(heatmapOriginal.shape) == 2:
        heatmapOriginal = cv2.cvtColor(heatmapOriginal, cv2.COLOR_GRAY2RGB)
    for (x, y), wall in pixel_to_wall.items():
        # Get the associated line
        #there are more than one wall projection for spatial cluster but they have the same value(?) why
        line = [h for h in walls_projections if h.spatial_cluster == wall.spatial_cluster][0]
        # Calculate the distance of the pixel from the line using the point-line distance formula
        distance = distance_point_line(line, x, y)
        if distance > max_distance :
            max_distance = distance
    colormap = plt.get_cmap('jet')
    colormap = colormap.reversed()
    heatmap = np.zeros_like(original_map, shape=(original_map.shape[0], original_map.shape[1], 3))
    print('max distance: {}'.format(max_distance))
    print('min distance: {}'.format(min_distance))
    for (x, y), wall in pixel_to_wall.items():
        line = [h for h in walls_projections if h.spatial_cluster == wall.spatial_cluster][0]
        distance = distance_point_line(line, x, y)
        if max_distance == 0 and min_distance == 0:
            normalized = 0
        else:
            normalized = (distance - min_distance) / (max_distance - min_distance)
        # Map the distance to a color in the colormap
        #normalized = distance/max_distance
        color = colormap(normalized)
        # Set the color in the heatmap
        heatmapOriginal[y, x] = (color[0]*255, color[1]*255, color[2]*255)
        heatmap[y, x] = (color[0]*255, color[1]*255, color[2]*255)
    plt.imsave(filepath + name + '.png', heatmap)
    plt.imsave(filepath + name + '_on_original.png', heatmapOriginal)
    return heatmap


def angular_distance(line1, line2):
        #√ertical line
        if line1.x1 == line1.x2:
            m1 = float('inf')
        else:
            m1 = (line1.y2 - line1.y1)/(line1.x2 - line1.x1)
        #vertical line
        if line2.x1 == line2.x2:
            m2 = float('inf')
        else:
            m2 = (line2.y2 - line2.y1)/(line2.x2 - line2.x1)
        #both vertical lines
        if m1 == float('inf') and m2 == float('inf'):
            return 0
        elif m1 == float('inf') and m2 != float('inf'):
            #angle between m2 line and horizontal line (0,0)
            alpha = abs(math.atan(m2))
            theta = np.pi/2 - alpha
            #convert to radians
            ris = theta
        elif m1 != float('inf') and m2 == float('inf'):
            #angle between m1 line and horizontal line (0,0)
            alpha = abs(math.atan(m1))
            theta = np.pi/2 - alpha
            #convert to radians
            ris = theta
        else:
            ris = math.atan(abs((m2 - m1) / (1 + m1 * m2)))
        return ris
"""
Calculate the angular heat map between walls and walls projections and draw it.
walls_projections: array of walls projections calculated in the Segment.set_weights() file.
pixel_to_wall:     a dictionary where each pixel that belongs to a wall gets associated with its wall.
                   {(x_1, y_1): wall), (x_2, y_2: wall). . .}
The heat map is constructed finding the angular distance between each wall and its wall_projection. All the pixels
belonging to a certain wall get assigned the same color.
Colors are given using the jet colormap (dark red means near, dark blue means distant), normalizing it based on the maximum distance
that has been computed using these pixels and walls.
"""
def angular_heatmap(walls_projections, pixel_to_wall, filepath, original_map, name):
    max_distance = 0
    min_distance = float('inf')
    # dictionary (wall : distance)
    distances = {}
    heatmapOriginal = original_map.copy()
    heatmapOriginal = cv2.cvtColor(heatmapOriginal, cv2.COLOR_GRAY2RGB)
    for _, wall in pixel_to_wall.items():
        wall_proj = [h for h in walls_projections if h.spatial_cluster == wall.spatial_cluster][0]
        distance = angular_distance(wall, wall_proj)
        distances[wall] = distance
        if distance < min_distance:
            min_distance = distance
        if distance > max_distance :
            max_distance = distance
    #print("Max distance: {}".format(max_distance))
    colormap = plt.get_cmap('jet')
    colormap = colormap.reversed()
    heatmap = np.zeros_like(original_map, shape=(original_map.shape[0], original_map.shape[1], 3))
    for (x, y), wall in pixel_to_wall.items():
        distance = distances[wall]
        # Map the distance to a color in the colormap
        #normalized = distance/max_distance
        normalized = (distance - min_distance) / (max_distance - min_distance)
        color = colormap(normalized)
        # Set the color in the heatmap
        heatmapOriginal[y, x] = (color[0]*255, color[1]*255, color[2]*255)
        heatmap[y, x] = (color[0]*255, color[1]*255, color[2]*255)
    plt.imsave(filepath + name + '.png', heatmap)
    plt.imsave(filepath + name + '_on_original.png', heatmapOriginal)
    return heatmap

#find the longest line that connects two walls, returns the coordinates
def find_longest_line(wall1, wall2):
    def distance(x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    distances = []
    distances.append([distance(wall1.x1, wall1.y1, wall2.x1, wall2.y1), wall1.x1, wall1.y1, wall2.x1, wall2.y1])
    distances.append([distance(wall1.x1, wall1.y1, wall2.x2, wall2.y2), wall1.x1, wall1.y1, wall2.x2, wall2.y2])
    distances.append([distance(wall1.x2, wall1.y2, wall2.x1, wall2.y1), wall1.x2, wall1.y2, wall2.x1, wall2.y1])
    distances.append([distance(wall1.x2, wall1.y2, wall2.x2, wall2.y2), wall1.x2, wall1.y2, wall2.x2, wall2.y2])
    return [x for x in distances if x[0] == max(distances, key=lambda x: x[0])[0]][0][1:]

# vertices is the output of layout.external_contour
def area_of_vertices(vertices):
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

# jaccard idx := |A∩B|/|AUB| where A∩B is the decluttered map contour area and AUB is the original map contour area
def jaccard_idx(map_binary, decluttered_map):
    # to work decluttered_map should have the pixels out of the contour in gray color, like the binary image of the original map.
    # go to fft_structure_extraction.py and make the decluttered_map have same color distribution as the original image:
    # light gray in background, black edges and white the inside of the indoor environment.

    screen_cnt_decl, vertices_decl = external_contour(decluttered_map)
    screen_cnt_original, vertices_original = external_contour(map_binary)
    dsg.draw_contour(vertices_original, 'original_contour', map_binary.shape)
    dsg.draw_contour(vertices_decl, 'decl_contour', map_binary.shape)
    area_decl = area_of_vertices(screen_cnt_decl)
    area_original = area_of_vertices(screen_cnt_original)
    return area_decl / area_original