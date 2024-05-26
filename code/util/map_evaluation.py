import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import object.Segment as Segment
import util.disegna as dsg
from random import randint
from object.Segment import radiant_inclination
from util.feature_matching import lines_of_direction
from skimage.metrics import structural_similarity
from sklearn.metrics import jaccard_score
import os
from util.disegna import draw_contour
from util.layout import external_contour
# our metric, when jaccard_idx = 1 it means we are not removing any part of the original image
# M = exp(alpha) * avg_dist
def map_metric(rose, jaccard_idx=1):
	dirs = [rose.param_obj.comp[0], rose.param_obj.comp[2]]
	angle = abs(dirs[0] - dirs[1])
	alpha = abs(np.pi/2 - angle)*180/np.pi
	avg_dist = avg_distance_walls_lines(rose.walls, rose.extended_segments, rose.original_binary_map, dirs)
	M = np.e**(alpha) * avg_dist / jaccard_idx
	return M

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

"""
walls: set of ExtendedSegment
lines: set of ExtendedSegment
dirs: main directions of lines
corrected_lines: if True dirs are manhattan and lines have been corrected so as to be manhattan
compute the average distance between the walls and the associated lines.

"""
def avg_distance_walls_lines(walls, lines, original_binary_map, dirs, corrected_lines=False):
	def distance_point_line(line, point_x, point_y):
		return np.abs(
			(line.y1 - line.y2) * point_x - (line.x1 - line.x2) * point_y + line.x1 * line.y2 - line.y1 * line.x2) / np.sqrt(
			(line.y1 - line.y2) ** 2 + (line.x1 - line.x2) ** 2
		)

	def closest_line_to_point(x, y, lines):
		cl_line = lines[0]
		cl_line_dist = distance_point_line(cl_line, x, y)
		for l in lines:
			dist = distance_point_line(l, x, y)
			if dist < cl_line_dist:
				cl_line_dist = dist
				cl_line = l
		return cl_line, cl_line_dist

	distances = []
	tol = 0.1
	#filepath = "/Users/gabrielesomaschini/Documents/ROSE2/ROSE2/declutter-reconstruct/code/"
	walls_without_out = remove_walls_outliers(walls, dirs)
	pixels_to_walls = pixel_to_wall(original_binary_map, walls_without_out)
	if corrected_lines:
		horiz_lines = lines_of_direction(lines, dirs[0])
		#dsg.draw_extended_lines(horiz_lines, walls, "horizontal_lines", original_binary_map.shape, filepath=filepath)
		vert_lines = lines_of_direction(lines, dirs[1])
		#dsg.draw_extended_lines(vert_lines, walls, "vertical_lines", original_binary_map.shape, filepath=filepath)
	for (x, y), wall in pixels_to_walls.items():
		# evaluate using manhattan directions and corrected lines
		if corrected_lines:
			wall_angle = radiant_inclination(wall.x1, wall.y1, wall.x2, wall.y2)
			if abs(wall_angle - dirs[0]) < tol:
				line, distance = closest_line_to_point(x, y, horiz_lines)
			else:
				line, distance = closest_line_to_point(x, y, vert_lines)
		else:
			#evaluate using the same map
			lines_spatial_clust = [l for l in lines if l.spatial_cluster == wall.spatial_cluster]
			line = lines_spatial_clust[0]
			distance = distance_point_line(line, x, y)
		distances.append(distance)
	return np.mean(distances)

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
				# white perchè nelle mappe create automaticamente quello è il valore che corrisponde al bianco.
				if (mask[y, x] == 255) and (original_binary_map[y, x] >= wall_white):
					pixel_to_wall[(x, y)] = wall
	return pixel_to_wall

#remove walls that have an inclination that is too much different from the main directions,
#this is decided by a threshold
def remove_walls_outliers(walls, dirs):
	tol = 0.2
	walls_without_outl = []
	for w in walls:
		angle = abs(radiant_inclination(w.x1, w.y1, w.x2, w.y2))
		if abs(angle - abs(dirs[0])) <= tol or abs(angle - abs(dirs[1])) <= tol:
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


#DOESN'T WORK YET.
"""
Calculate the angular heat map between walls and walls projections and draw it.
walls_projections: array of walls projections calculated in the Segment.set_weights() file.
pixel_to_wall:     a dictionary where each pixel that belongs to a wall gets associated with its wall.
  				   {(x_1, y_1): wall), (x_2, y_2: wall). . .}
The heat map is constructed finding the angular distance between each wall cluster and its wall projection, 
each wall cluster gets a distance summing the distance between each wall and its projection. All the pixels
belonging to a certain wall get assigned the same color.
Colors are given using the jet colormap (dark red means near, dark blue means distant), normalizing it based on the maximum distance
that has been computed using these pixels and walls.
"""
def angular_heatmap_cluster(walls_projections, pixel_to_wall, filepath, original_map, name):
	max_distance = 0
	# dictionary (wall : distance)
	distances = {}
	heatmapOriginal = original_map.copy()
	heatmapOriginal = cv2.cvtColor(heatmapOriginal, cv2.COLOR_GRAY2RGB)
	walls = set(pixel_to_wall.values())
	# WALLS WITH THE SAME WALL_CLUSTER
	wall_clusters = {}
	lines = []
	for w1 in walls:
		if w1.wall_cluster not in wall_clusters:
			wall_clusters[w1.wall_cluster] = [w1]
		for w2 in walls:
			if w1 != w2 and w1.wall_cluster == w2.wall_cluster:
				wall_clusters[w1.wall_cluster].append(w2)
	for w_clust, walls in wall_clusters.items():
		#da sistemare è wall cluster non spatial cluster
		wall_proj = [h for h in walls_projections if h.spatial_cluster == walls[0].spatial_cluster][0]
		wall_cluster_length = 0
		# calculate wall_cluster length
		for wall1 in walls:
			for wall2 in walls:
				if wall1 != wall2:

					wall1_length = Segment.length(wall1.x1, wall1.y1, wall1.x2, wall1.y2)
					wall2_length = Segment.length(wall2.x1, wall2.y1, wall2.x2, wall2.y2)
					wall_distance = Segment.segments_distance(wall1.x1, wall1.y1, wall1.x2, wall1.y2,
															  wall2.x1, wall2.y1, wall2.x2,
															  wall2.y2) + wall1_length + wall2_length
					#wall_distance = max(wall1_length, wall2_length, wall_distance)
					if wall_distance > wall_cluster_length:
						x1, y1, x2, y2 = find_longest_line(wall1, wall2)
						angle = Segment.radiant_inclination(x1, y1, x2, y2)
						if angle > 0.3:
							x1, y1, x2, y2 = wall1.x1, wall1.y1, wall1.x2, wall1.y2
							wall_cluster_length = 0
						else:
							wall_cluster_length = wall_distance
						line = Segment.Segment(x1, y1, x2, y2)
		if wall_cluster_length == 0:
			w = walls[0]
			line = Segment.Segment(w.x1, w.y1, w.x2, w.y2)
		lines.append(line)
		distance = angular_distance(line, wall_proj)
		for wall in walls:
			distances[wall] = distance
		if distance > max_distance:
			max_distance = distance
	""""
	WEIGHT WALLS ANGULAR DISTANCE BASED ON HOW MUCH WALL CLUSTER THEY OCCUPY.
	DOESN'T WORK BECAUSE WALL CLUSTER LENGTH DOESN'T COUNT ALSO THE FACT THAT THERE ARE MULTIPLE
	WALLS PARALLEL
	for idx, w in enumerate(wall_clusters):
		lines = []
		distance = 0
		wall_proj = [h for h in walls_projections if h.spatial_cluster == w[0].spatial_cluster][0]
		wall_cluster_length = 0
		#calculate wall_cluster length
		for wall1 in wall_clusters[idx]:
			for wall2 in wall_clusters[idx]:
				if wall1 != wall2:
					wall1_length = Segment.length(wall1.x1, wall1.y1, wall1.x2, wall1.y2)
					wall2_length = Segment.length(wall2.x1, wall2.y1, wall2.x2, wall2.y2)
					wall_distance = Segment.segments_distance(wall1.x1, wall1.y1, wall1.x2, wall1.y2,
																 wall2.x1, wall2.y1, wall2.x2, wall2.y2) + wall1_length + wall2_length
					if wall_distance > wall_cluster_length:
						line = (wall1.x1, wall1.y1, wall2.x2, wall2.y2)
						wall_cluster = wall1.spatial_cluster
						wall_cluster_length = wall_distance
		if wall_cluster_length == 0:
			w = wall_clusters[idx][0]
			wall_cluster = w.spatial_cluster
			wall_cluster_length = Segment.length(w.x1, w.y1, w.x2, w.y2)
		print("Wall cluster length: {}".format(wall_cluster_length))
		# for each wall find the distance between wall and wall projection and weight it based on
		# the percentage of wall that covers its wall_cluster
		for wall in wall_clusters[idx]:
			wall_length = Segment.length(wall.x1, wall.y1, wall.x2, wall.y2)
			#percentuale di wall_cluster occupato dal wall
			#NON VA BENE PERCHE' ALCUNE PARETI NON SI TROVANO SULLA STESSA RETTA DEL WALL CLUSTER QUINDI
			# LA PERCENTUALE TOTALE VIENE > 1
			perc_wall = wall_length/wall_cluster_length
			print("Wall percentage: {}".format(perc_wall))
			distance += (angular_distance(wall, wall_proj) * perc_wall)
		print("Distance: {}".format(distance))
		for wall in wall_clusters[idx]:
			distances[wall] = distance
		if distance > max_distance:
			max_distance = distance
		"""
	colormap = plt.get_cmap('jet')
	colormap = colormap.reversed()
	heatmap = np.zeros_like(original_map, shape=(original_map.shape[0], original_map.shape[1], 3))
	for (x, y), wall in pixel_to_wall.items():
		distance = distances[wall]
		# Map the distance to a color in the colormap
		normalized = distance / max_distance
		color = colormap(normalized)
		# Set the color in the heatmap
		heatmapOriginal[y, x] = (color[0] * 255, color[1] * 255, color[2] * 255)
		heatmap[y, x] = (color[0] * 255, color[1] * 255, color[2] * 255)
	plt.imsave(filepath + name + '.png', heatmap)
	plt.imsave(filepath + name + '_on_original.png', heatmapOriginal)
	dsg.draw_walls(lines, "lines", original_map.shape, filepath=filepath)
	return heatmap