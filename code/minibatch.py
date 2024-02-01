from __future__ import division

import os
import time

import cv2
import util.disegna as dsg
import util.layout as lay
import util.evaluation as eval
import util.postprocessing as post
import util.voronoi as vr
from object import Segment as sg, ExtendedSegment
from object import Surface as Surface
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot as plt
import numpy as np
import FFT_MQ as fft
import util.map_evaluation as mp

def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def print_parameters(param_obj, path_obj):
    file_parameter = open(path_obj.filepath + 'parameters.txt', 'w')
    param = param_obj.__dict__
    for par in param:
        file_parameter.write(par + ": %s\n" % (getattr(param_obj, par)))


def final_routine(img_ini, param_obj, size, draw, extended_segments_th1_merged, ind, rooms_th1, filepath):
    #post.clear_rooms(filepath + '8b_rooms_th1.png', self.param_obj, rooms_th1)
    if draw.rooms_on_map:
        segmentation_map_path = dsg.draw_rooms_on_map(img_ini, '8b_rooms_th1_on_map', size, filepath=filepath)
    """
    if draw.rooms_on_map_prediction:
        dsg.draw_rooms_on_map_prediction(img_ini, '8b_rooms_th1_on_map_prediction', size, filepath=filepath)

    if draw.rooms_on_map_lines:
        dsg.draw_rooms_on_map_plus_lines(img_ini, self.extended_segments_th1_merged, '8b_rooms_th1_on_map_th1' + str(ind), size,
                                         filepath=filepath)
    """
    # -------------------------------------POST-PROCESSING------------------------------------------------

    segmentation_map_path_post, colors = post.oversegmentation(segmentation_map_path, self.param_obj.th_post, filepath=filepath)
    return segmentation_map_path_post, colors

# map1 - map2
# maps have to be binary (black and white) and of the same size
def map_difference(map1, map2):
    result = np.zeros(map1.shape)
    for i in range(map1.shape[0]):
        for j in range(map1.shape[1]):
            if(map1[i,j] == 255 and map2[i,j] == 0):
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result
class Minibatch:
    def start_main(self, par, param_obj, path_obj):
        self.param_obj = param_obj
        self.param_obj.tab_comparison = [[''], ['precision_micro'], ['precision_macro'], ['recall_micro'], ['recall_macro'],
                                    ['iou_micro_mean_seg_to_gt'], ['iou_macro_seg_to_gt'], ['iou_micro_mean_gt_to_seg'],
                                    ['iou_macro_gt_to_seg']]
        start_time_main = time.time()

        draw = par.ParameterDraw()
        self.filepath = path_obj.filepath
        print(self.filepath)
        print_parameters(self.param_obj, path_obj)

        # ----------------------------1.0_LAYOUT OF ROOMS------------------------------------
        # ------ starting layout
        # read the original image
        orebro_img = cv2.imread(path_obj.orebro_img)
        width = orebro_img.shape[1]
        height = orebro_img.shape[0]
        self.size = [width, height]
        img_rgb = cv2.bitwise_not(orebro_img)
        # making a copy of original image
        img_ini = cv2.imread(path_obj.metric_map_path)
        self.original_map = img_ini.copy()
        # -------------------------------------------------------------------------------------

        # -----------------------------1.1_CANNY AND HOUGH-------------------------------------

        self.walls, canny = lay.start_canny_and_hough(img_rgb, self.param_obj)

        print("walls:", len(self.walls))

        if draw.map:
            dsg.draw_image(img_ini, '0_Map', self.size, filepath=self.filepath)

        if draw.hough:
            dsg.draw_hough(img_ini, self.walls, '2_Hough', self.size, filepath=self.filepath)

        lines = self.walls
        self.walls = lay.create_walls(lines)
        # BINARY MAP
        self.original_binary_map = cv2.bitwise_not(img_ini)
        self.original_binary_map = cv2.cvtColor(self.original_binary_map, cv2.COLOR_RGB2GRAY)
        plt.imsave(self.filepath + "binary_map.png", self.original_binary_map)
        if not self.param_obj.bormann:
            if draw.walls:
                # draw Segments
                dsg.draw_walls(self.walls, '3_Walls', self.size, filepath=self.filepath)
            lim1, lim2 = 300, 450
            """""
            while not(lim1 <= len(walls) <= lim2):
                if self.param_obj.filter_level <= 0.12:
                    break
    
                if len(walls) < lim1:
                    self.param_obj.set_filter_level(self.param_obj.filter_level - 0.02)
                if len(walls) > lim2:
                    self.param_obj.set_filter_level(self.param_obj.filter_level + 0.02)
                fft.main(path_obj.metric_map_path, path_obj.path_orebro, self.param_obj.filter_level, self.param_obj)
                path_obj.orebro_img = filepath + 'OREBRO_' + str(self.param_obj.filter_level) + '.png'
    
                # ----------------------------1.0_LAYOUT OF ROOMS------------------------------------
                # ------ starting layout
                # read the original image
                orebro_img = cv2.imread(path_obj.orebro_img)
                width = orebro_img.shape[1]
                height = orebro_img.shape[0]
                size = [width, height]
                img_rgb = cv2.bitwise_not(orebro_img)
                # making a copy of original image
                img_ini = cv2.imread(path_obj.metric_map_path)
    
                # -------------------------------------------------------------------------------------
    
                # -----------------------------1.1_CANNY AND HOUGH-------------------------------------
    
                walls, canny = lay.start_canny_and_hough(img_rgb, self.param_obj)
    
                print("walls:", len(walls))
    
                if draw.map:
                    dsg.draw_image(img_ini, '0_Map', size, filepath=filepath)
    
                if draw.hough:
                    dsg.draw_hough(img_ini, walls, '2_Hough', size, filepath=filepath)
    
                lines = walls
                walls = lay.create_walls(lines)
                print("lines:", len(lines), 'walls:', len(walls))
    
                if draw.walls:
                    # draw Segments
                    dsg.draw_walls(walls, '3_Walls', size, filepath=filepath)
            """


        # ------------1.2_SET XMIN, YMIN, XMAX, YMAX OF walls-----------------------------------
        # from all points of walls select x and y coordinates max and min.
        extremes = sg.find_extremes(self.walls)
        xmin = extremes[0]
        xmax = extremes[1]
        ymin = extremes[2]
        ymax = extremes[3]
        offset = self.param_obj.offset
        xmin -= offset
        xmax += offset
        ymin -= offset
        ymax += offset

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > self.size[0]:
            xmax = self.size[0]
        if ymax > self.size[1]:
            ymax = self.size[1]

        # -------------------------------------------------------------------------------------

        # ---------------1.3 EXTERNAL CONTOUR--------------------------------------------------

        img_cont = cv2.imread(path_obj.metric_map_path)
        (contours, self.vertices) = lay.external_contour(img_cont)

        if draw.contour:
            dsg.draw_contour(self.vertices, '4_Contour', self.size, filepath=self.filepath)

        # -------------------------------------------------------------------------------------

        # ---------------1.4_MEAN SHIFT TO FIND ANGULAR CLUSTERS-------------------------------

        indexes, self.walls, angular_clusters = lay.cluster_ang(self.param_obj.h, self.param_obj.minOffset, self.walls, diagonals=self.param_obj.diagonals)
        angular_clusters = lay.assign_orebro_direction(self.param_obj.comp, self.walls)
        if draw.angular_cluster:
            dsg.draw_angular_clusters(angular_clusters, self.walls, '5a_angular_clusters', self.size, filepath=self.filepath)
        # -------------------------------------------------------------------------------------

        # ---------------1.5_SPATIAL CLUSTERS--------------------------------------------------

        # TODO Valerio's method
        wall_clusters = lay.get_wall_clusters(self.walls, angular_clusters)

        wall_cluster_without_outliers = []
        for cluster in wall_clusters:
            if cluster != -1:
                wall_cluster_without_outliers.append(cluster)

        # now that I have a list of clusters related to walls, I want to merge those very close each other
        # obtain representatives of clusters (all except outliers)
        representatives_segments = lay.get_representatives(self.walls, wall_cluster_without_outliers)

        if draw.representative_segments:
            dsg.draw_representative_segments(representatives_segments, "5b_representative_segments", self.size, filepath=self.filepath)

        representatives_segments = sg.spatial_clustering(self.param_obj.sogliaLateraleClusterMura, representatives_segments)

        # now we have a set of Segments with correct spatial cluster, now set the others with same wall_cluster
        spatial_clusters = lay.new_spatial_cluster(self.walls, representatives_segments, self.param_obj)
        if draw.spatial_wall_cluster:
            dsg.draw_spatial_wall_clusters(wall_clusters, self.walls, '5c_spatial_wall_cluster', self.size, filepath=self.filepath)

        if draw.spatial_cluster:
            dsg.draw_spatial_clusters(spatial_clusters, self.walls, '5d_spatial_clusters', self.size, filepath=self.filepath)

        # -------------------------------------------------------------------------------------

        # ------------------------1.6 EXTENDED_LINES-------------------------------------------

        (extended_lines, self.extended_segments) = lay.extend_line(spatial_clusters, self.walls, xmin, xmax, ymin, ymax)

        if draw.extended_lines:
            make_folder(self.filepath, 'Extended_Lines')
            dsg.draw_extended_lines(self.extended_segments, self.walls, '7a_extended_lines', self.size, filepath=self.filepath + '/Extended_Lines')
            # Create an empty mask with the same dimensions as the image
        self.extended_segments, self.walls_projections = sg.set_weights(self.extended_segments, self.walls)
        
        # this is used to merge together the extended_segments that are very close each other.
        extended_segments_merged = ExtendedSegment.merge_together(self.extended_segments, self.param_obj.distance_extended_segment, self.walls)
        extended_segments_merged, walls_projections_merged = sg.set_weights(extended_segments_merged, self.walls)

        """"
        dsg.draw_hough_black(self.original_binary_map, lines, 'removedBadPixelsOriginal', size, filepath=filepath)
        dsg.draw_hough_black(orebro_img, lines, 'removedBadPixelsClean', size, filepath=filepath)
        correctedOriginal = plt.imread(filepath + 'removedBadPixelsOriginal.png')
        correctedClean = plt.imread(filepath + 'removedBadPixelsClean.png')
        correctedOriginal = cv2.cvtColor(correctedOriginal, cv2.COLOR_RGB2GRAY)
        correctedClean = cv2.cvtColor(correctedClean, cv2.COLOR_RGB2GRAY)
        dsg.draw_hough_segment_white(correctedOriginal, walls_projections, 'correctedOriginal', size, filepath=filepath)
        dsg.draw_hough_segment_white(correctedClean, walls_projections, 'correctedClean', size, filepath=filepath)
        """

        # this is needed in order to maintain the extended lines of the offset STANDARD
        border_lines = lay.set_weight_offset(extended_segments_merged, xmax, xmin, ymax, ymin)
        self.extended_segments_th1_merged, ex_li_removed = sg.remove_less_representatives(extended_segments_merged, self.param_obj.th1)
        if draw.extended_lines:
            dsg.draw_extended_lines(extended_segments_merged, self.walls, '7a_extended_lines_merged', self.size,
                                    filepath=self.filepath + '/Extended_Lines')
            dsg.draw_extended_lines(self.extended_segments_th1_merged, self.walls, '7a_extended_lines_th1_merged', self.size, filepath=self.filepath + '/Extended_Lines')
        lis = []
        for line in ex_li_removed:
            short_line = sg.create_short_ex_lines(line, self.walls, self.size, self.extended_segments_th1_merged)
            if short_line is not None:
                lis.append(short_line)

        lis, _ = sg.set_weights(lis, self.walls)
        lis, _ = sg.remove_less_representatives(lis, 0.1)
        for el in lis:
            self.extended_segments_th1_merged.append(el)
        
        if draw.extended_lines:
            dsg.draw_extended_lines(self.extended_segments_th1_merged, self.walls, '7a_extended_lines_merged_plus_small', self.size, filepath=self.filepath + '/Extended_Lines')


        if not self.param_obj.stop_after_lines:
            # DRAW HEATMAP OF WALL PROJECTIONS
            mp.distance_heat_map(self.walls_projections, pixel_to_wall, filepath, img_ini, 'distance_heatmap')
            mp.angular_heatmap(self.walls_projections, pixel_to_wall, filepath, self.original_binary_map, 'angular_heatmap')
            # mp.angular_heatmap_cluster(walls_projections, pixel_to_wall, filepath, self.original_binary_map, 'angular_heatmap_cluster')
            dsg.draw_walls(self.walls_projections, "wall_projections", size, filepath=filepath)

            # -------------------------------------------------------------------------------------

            # --------------------------------1.7_EDGES--------------------------------------------

            # creating edges as intersection between extended lines

            edges = sg.create_edges(self.extended_segments)
            edges_th1 = sg.create_edges(self.extended_segments_th1_merged)
            # sg.set_weight_offset_edges(border_lines, edges_th1)

            # -------------------------------------------------------------------------------------

            # ---------------------------1.8_SET EDGES WEIGHTS-------------------------------------

            edges, _ = sg.set_weights(edges, walls)
            edges_th1, _ = sg.set_weights(edges_th1, walls)

            if draw.edges:
                make_folder(filepath, 'Edges')
                dsg.draw_edges(edges, walls, -1, '7c_edges', size, filepath=filepath + '/Edges')
                dsg.draw_edges(edges, walls, self.param_obj.threshold_edges, '7c_edges_weighted', size, filepath=filepath + '/Edges')
                dsg.draw_edges(edges_th1, walls, -1, '7c_edges_th1', size, filepath=filepath + '/Edges')
                dsg.draw_edges(edges_th1, walls, self.param_obj.threshold_edges, '7c_edges_th1_weighted', size, filepath=filepath + '/Edges')
            # -------------------------------------------------------------------------------------

            # ----------------------------1.9_CREATE CELLS-----------------------------------------

            cells_th1 = Surface.create_cells(edges_th1)

            # -------------------------------------------------------------------------------------

            # ----------------Classification of Facces CELLE-----------------------------------------------------

            # Identification of Cells/Faces that are Inside or Outside the map
            global centroid
            if par.metodo_classificazione_celle == 1:
                print("1.classification method: ", par.metodo_classificazione_celle)
                (cells_th1, cells_out_th1, cells_polygons_th1, indexes_th1, cells_partials_th1, contour_th1, centroid_th1, points_th1) = lay.classification_surface(self.vertices, cells_th1, self.param_obj.division_threshold)

            # -------------------------------------------------------------------------------------

            # ---------------------------POLYGON CELLS---------------------------------------------
            # TODO this method could be deleted. check. Not used anymore.
            (cells_polygons_th1, polygon_out_th1, polygon_partial_th1, centroid_th1) = lay.create_polygon(cells_th1, cells_out_th1,cells_partials_th1)

            if draw.cells_in_out:
                dsg.draw_cells(cells_polygons_th1, polygon_out_th1, polygon_partial_th1, '8a_cells_in_out_partial_th1', size, filepath=filepath)

            # ----------------------MATRICES L, D, D^-1, ED M = D^-1 * L--------------------------

            (matrix_l_th1, matrix_d_th1, matrix_d_inv_th1, X_th1) = lay.create_matrices(cells_th1, sigma=self.param_obj.sigma)

            # -------------------------------------------------------------------------------------

            # ----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-------------------------

            cluster_cells_th1 = lay.DB_scan(self.param_obj.eps, self.param_obj.minPts, X_th1, cells_polygons_th1)

            if draw.dbscan:
                colors_th1, fig, ax = dsg.draw_dbscan(cluster_cells_th1, cells_th1, cells_polygons_th1, edges_th1, contours, '7b_DBSCAN_th1', size, filepath=filepath)

            # -------------------------------------------------------------------------------------

            # ----------------------------POLYGON ROOMS--------------------------------------------

            rooms_th1, spaces_th1 = lay.create_space(cluster_cells_th1, cells_th1, cells_polygons_th1)

            # -------------------------------------------------------------------------------------

            # searching partial cells
            border_coordinates = [xmin, ymin, xmax, ymax]
            # TODO check how many time is computed
            cells_partials_th1, polygon_partial_th1 = lay.get_partial_cells(cells_th1, cells_out_th1, border_coordinates)

            polygon_out_th1 = lay.get_out_polygons(cells_out_th1)

            if draw.sides:
                dsg.draw_sides(edges_th1, '14a_sides_th1', size, filepath=filepath)
            if draw.rooms:
                fig, ax, patches = dsg.draw_rooms(rooms_th1, colors_th1, '8b_rooms_th1', size, filepath=filepath)

            # ---------------------------------END LAYOUT------------------------------------------

            ind = 0

            segmentation_map_path_post, colors = final_routine( img_ini, self.param_obj, size, draw,
                                                               self.extended_segments_th1_merged, ind, rooms_th1, filepath=filepath)
            old_colors = []
            voronoi_graph, coordinates = vr.compute_voronoi_graph(path_obj.metric_map_path, self.param_obj,
                                                                  False, '', self.param_obj.bormann, filepath=filepath)
            while old_colors != colors and ind < self.param_obj.iterations:
                ind += 1
                old_colors = colors
                vr.voronoi_segmentation(patches, colors_th1, size, voronoi_graph, coordinates, self.param_obj.comp, path_obj.metric_map_path, ind, filepath=filepath)
                segmentation_map_path_post, colors = final_routine(img_ini, self.param_obj, size, draw,
                                                                   self.extended_segments_th1_merged, ind, rooms_th1, filepath=filepath)
            rooms_th1 = make_rooms(patches)
            colors_final = []
            print(len(rooms_th1))
            for r in rooms_th1:
                colors_final.append(0)
            dsg.draw_rooms(rooms_th1, colors_final, '8b_rooms_th1_final', size, filepath=filepath)
            # -------------------------------------------------------------------------------------

            end_time_main = time.time()

            # write the times on file
            time_file = filepath + '17_times.txt'

            with open(time_file, 'w+') as TIMEFILE:
                print("time for main is: ", end_time_main - start_time_main, file=TIMEFILE)

def make_rooms(patches):
    l = []
    for p in patches:
        l.append(Polygon(p.get_path().vertices))
    return l

