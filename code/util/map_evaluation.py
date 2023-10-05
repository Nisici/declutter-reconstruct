import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import object.Segment as Segment

"""
Calculate the distance heat map between walls and walls projections and draw it.
walls_projections: array of walls projections calculated in the Segment.set_weights() file.
pixel_to_wall:     a dictionary where each pixel that belongs to a wall gets associated with its wall.
  				   {(x_1, y_1): wall), (x_2, y_2: wall). . .}
the heat map is constructed finding the distance between each pixel and its wall_projection, colors are given
using the jet colormap (dark red means near, dark blue means distant), normalizing it based on the maximum distance
that has been computed using these pixels and walls.
"""
def distance_heat_map(walls_projections, pixel_to_wall, filepath, original_map, name):
	def distance_point_line(line, point_x, point_y):
		return np.abs(
			(line.y1 - line.y2) * point_x - (line.x1 - line.x2) * point_y + line.x1 * line.y2 - line.y1 * line.x2) / np.sqrt(
			(line.y1 - line.y2) ** 2 + (line.x1 - line.x2) ** 2
		)
	max_distance = 0
	heatmapOriginal = original_map.copy()
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
	for (x, y), wall in pixel_to_wall.items():
		line = [h for h in walls_projections if h.spatial_cluster == wall.spatial_cluster][0]
		distance = distance_point_line(line, x, y)
		# Map the distance to a color in the colormap
		normalized = distance/max_distance
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
	# dictionary (wall : distance)
	distances = {}
	heatmapOriginal = original_map.copy()
	heatmapOriginal = cv2.cvtColor(heatmapOriginal, cv2.COLOR_GRAY2RGB)
	for _, wall in pixel_to_wall.items():
		wall_proj = [h for h in walls_projections if h.spatial_cluster == wall.spatial_cluster][0]
		distance = angular_distance(wall, wall_proj)
		distances[wall] = distance
		if distance > max_distance :
			max_distance = distance
	colormap = plt.get_cmap('jet')
	colormap = colormap.reversed()
	heatmap = np.zeros_like(original_map, shape=(original_map.shape[0], original_map.shape[1], 3))
	for (x, y), wall in pixel_to_wall.items():
		distance = distances[wall]
		# Map the distance to a color in the colormap
		normalized = distance/max_distance
		color = colormap(normalized)
		# Set the color in the heatmap
		heatmapOriginal[y, x] = (color[0]*255, color[1]*255, color[2]*255)
		heatmap[y, x] = (color[0]*255, color[1]*255, color[2]*255)
	plt.imsave(filepath + name + '.png', heatmap)
	plt.imsave(filepath + name + '_on_original.png', heatmapOriginal)
	return heatmap


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
	wall_clusters = []
	idx = 0
	for w1 in walls:
		wall_clusters.append([w1])
		for w2 in walls:
			if w1 != w2 and w1.wall_cluster == w2.wall_cluster:
				wall_clusters[idx].append(w2)
		idx+=1
	for idx, w in enumerate(wall_clusters):
		distance = 0
		wall_proj = [h for h in walls_projections if h.spatial_cluster == w[0].spatial_cluster][0]
		for wall in wall_clusters[idx]:
			distance += angular_distance(wall, wall_proj)
		for wall in wall_clusters[idx]:
			distances[wall] = distance
		if distance > max_distance:
			max_distance = distance
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
	return heatmap