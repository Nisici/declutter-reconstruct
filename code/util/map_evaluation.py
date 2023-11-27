import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import object.Segment as Segment
import util.disegna as dsg
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