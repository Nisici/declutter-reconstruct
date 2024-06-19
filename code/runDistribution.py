import glob
import os
import shutil
import datetime
import tabulate
import numpy as np

import parameters as par
import minibatch
import re
import FFT_MQ as fft
from util.map_evaluation import walls_distance_distribution, walls_directions_distribution, pixels_walls_distance_distribution
from util.feature_correction import correct_lines, manhattan_directions
from object.Segment import radiant_inclination
from util.disegna import draw_extended_lines
from object.Segment import segments_distance
import PIL.Image
import os
from matplotlib import pyplot as plt
import pyemd
from scipy.spatial import distance_matrix, distance
from scipy.stats import wasserstein_distance
import util.disegna as dsg

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def check_int(name_folder):
    maps = os.listdir(name_folder)
    while True:
        index = 0
        for image in maps:
            print(str(index) + '   ' + image)
            index += 1
        try:
            val = int(input('insert the number you want to select\n'))
            if 0 <= val < index:
                return maps[val]
        except ValueError:
            print('invalid input')


def check_action():
    while True:
        print('What kind of action do you want to do?\n1     batch\n2     single action\n3     sliding parameters')
        try:
            val = int(input('insert the number of action selected\n'))
            if val == 1:
                return 'batch'
            elif val == 2:
                return 'single action'
            elif val == 3:
                return 'sliding parameters'
        except ValueError:
            print('invalid input')


def sort_name(par):
    return par[0]

def rose_single_action():
    # ----------------PARAMETERS OBJECTS------------------------
    # loading parameters from parameters.py
    parameters_object = par.ParameterObj()

    # loading path object with all path and name interesting for code
    paths = par.PathObj()

    # ----------------------------------------------------------

    # taking all the folders inside the path INPUT/IMGs
    list_dir = './data/INPUT/IMGs'
    # asking the user what folder want to use
    t = list_dir + '/' + check_int(list_dir)
    # asking the user what folder want to use
    paths.name_folder_input = check_int(t)
    paths.path_folder_input = t + '/' + paths.name_folder_input
    if paths.name_folder_input == 'Bormann' or paths.name_folder_input == 'Bormann_furnitures':
        parameters_object.bormann = True
    else:
        parameters_object.bormann = False

    # asking if is a batch action or single action
    action = check_action()

    # asking what map if is a single action
    if action == 'single action':
        # saving the output folder where the output is saved
        make_folder('data/OUTPUT', 'SINGLEMAP')
        paths.path_folder_output = './data/OUTPUT/SINGLEMAP'
        # asking what map to use
        paths.metric_map_name = check_int(paths.path_folder_input)
        paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
    running_time = str(datetime.datetime.now())[:-7].replace(' ', '@')
    # ----------------------------------------------------------------
    # making the log folder
    paths.path_log_folder = os.path.join(paths.path_folder_output, paths.metric_map_name[:-4])
    make_folder(paths.path_folder_output, paths.metric_map_name[:-4])
    make_folder(paths.path_log_folder, running_time)
    paths.filepath = paths.path_log_folder + '/' + running_time + '/'
    # copying the parameters file
    shutil.copy('parameters.py', paths.path_log_folder + '/' + running_time + '/parameters.py')
    # ----------------------------------------------------------------
    # orebro
    make_folder(paths.path_log_folder, running_time + '/OREBRO')
    paths.path_orebro = paths.path_log_folder + '/' + running_time + '/OREBRO'
    paths.orebro_img = paths.filepath + 'OREBRO_' + str(parameters_object.filter_level) + '.png'
    fft.main(paths.metric_map_path, paths.path_orebro, parameters_object.filter_level, parameters_object)
    # ----------------------------------------------------------------
    # evaluation
    paths.gt_color = './data/INPUT/gt_colored/' + paths.name_folder_input + '/' + paths.metric_map_name
    # ----------------------------------------------------------------
    # starting main
    print('map name ', paths.metric_map_name)
    m = minibatch.Minibatch()
    m.start_main(par, parameters_object, paths)
    return m, parameters_object

#returns union distribution of pixels wall distances to lines and wall distances to corrected lines
def rose_walls_distances_distributions(rose, par):
    dirs = [par.comp[0], par.comp[2]]
    corrected_lines = correct_lines(rose.extended_segments, manhattan_directions(dirs))
    distrib_walls_dist = pixels_walls_distance_distribution(rose.walls, rose.extended_segments, rose.original_binary_map)
    distrib_walls_dist_corrected_lines = pixels_walls_distance_distribution(rose.walls, corrected_lines, rose.original_binary_map)
    for k in distrib_walls_dist_corrected_lines.keys():
        if k not in distrib_walls_dist.keys():
            distrib_walls_dist[k] = 0
    for k in distrib_walls_dist.keys():
        if k not in distrib_walls_dist_corrected_lines.keys():
            distrib_walls_dist_corrected_lines[k] = 0
    return distrib_walls_dist, distrib_walls_dist_corrected_lines

#distance between walls and corrected lines
def rose_walls_corrected_distances_distributions(rose, par):
    dirs = [par.comp[0], par.comp[2]]
    manhattan_dirs = manhattan_directions(dirs)
    corrected_walls = correct_lines(rose.walls, manhattan_dirs)
    corrected_lines = correct_lines(rose.extended_segments, manhattan_dirs)
    distrib_walls_dist = pixels_walls_distance_distribution(corrected_walls, rose.extended_segments)
    distrib_walls_dist_corrected_lines = pixels_walls_distance_distribution(corrected_walls, corrected_lines)
    for k in distrib_walls_dist_corrected_lines.keys():
        if k not in distrib_walls_dist.keys():
            distrib_walls_dist[k] = 0
    for k in distrib_walls_dist.keys():
        if k not in distrib_walls_dist_corrected_lines.keys():
            distrib_walls_dist_corrected_lines[k] = 0
    return distrib_walls_dist, distrib_walls_dist_corrected_lines

def rose_walls_angular_distributions(rose, par):
    dirs = [par.comp[0], par.comp[2]]
    manhattan_dirs = manhattan_directions(dirs)
    corrected_walls = correct_lines(rose.walls, manhattan_dirs)
    dsg.draw_walls(corrected_walls, "corrected-walls", rose.size, filepath=rose.filepath)
    distrib_walls_angular = walls_directions_distribution(rose.walls, rose.original_binary_map)
    distrib_walls_corrected_angular = walls_directions_distribution(corrected_walls, rose.original_binary_map)
    for k in distrib_walls_corrected_angular.keys():
        if k not in distrib_walls_angular.keys():
            distrib_walls_angular[k] = 0
    for k in distrib_walls_angular.keys():
        if k not in distrib_walls_corrected_angular.keys():
            distrib_walls_corrected_angular[k] = 0
    return distrib_walls_angular, distrib_walls_corrected_angular

def save_distribution_plot(union_distrib, filepath, name=""):
    plt.close()
    plt.figure()
    keys = list(union_distrib.keys())
    values = list(union_distrib.values())
    plt.bar(keys, values, width=0.2)
    plt.xlabel("Distance")
    plt.ylabel("Number of walls")
    plt.savefig(os.path.join(filepath, name + "histogram.png"))
    plt.close()

def make_distance_matrix(dict1, dict2, metric='euclidean'):
    dist_matrix = np.zeros((len(dict1), len(dict1)), dtype='float64')
    i = 0
    j = 0
    # give weights based on |k1 - k2|
    for k1, v1 in dict1.items():
        for k2, v2 in dict2.items():
            if metric == 'angular':
                # L1 because small values shouldn't become smaller like in L2
                dist_matrix[i, j] = min(abs(k1 - k2), abs(k1 - 3.14 - k2), abs(k1 + 3.14 - k2))
                #dist_matrix[i, j] = abs(k1 - k2)
            elif metric == 'euclidean':
                dist_matrix[i, j] = abs(k1 - k2) ** 2
            j += 1
        j = 0
        i += 1
    #vals1 = np.reshape(list(dict1.values()), (-1, 1))
    #vals2 = np.reshape(list(dict2.values()), (-1, 1))
    #dist_matrix *= distance.cdist(vals1, vals2, metric)
    return dist_matrix

#calculate EDM between two distributions in form of dictionaries
def EMD(dict1, dict2):
    print("Num pixels dict1: {}".format(sum(dict1.values())))
    print("Num pixels dict2: {}".format(sum(dict2.values())))
    num_of_pixels = sum(dict1.values())
    dist_matrix = make_distance_matrix(dict1, dict2, metric='euclidean')
    distrib_walls_lines = np.reshape(list(dict1.values()), (-1, 1))
    distrib_walls_corrected_lines = np.reshape(list(dict2.values()), (-1, 1))
    distrib_walls_lines = distrib_walls_lines.flatten().astype('float64')
    distrib_walls_corrected_lines = distrib_walls_corrected_lines.flatten().astype('float64')
    emd, flow = pyemd.emd_with_flow(distrib_walls_lines, distrib_walls_corrected_lines, dist_matrix)
    #flow = np.array(flow).flatten()
    #print("Number of pixels moved: {}".format(len(flow)))
    emd = (emd**0.5) / num_of_pixels
    return emd

def EMD_angular(dict1, dict2):
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

def main():
    rose1, par1 = rose_single_action()
    #calculate distributions
    distrib_walls_lines, distrib_walls_corrected_lines = rose_walls_distances_distributions(rose1, par1)
    distrib_walls_angular, distrib_walls_angular_corrected_lines = rose_walls_angular_distributions(rose1, par1)
    #save plots
    save_distribution_plot(distrib_walls_lines, rose1.filepath)
    save_distribution_plot(distrib_walls_corrected_lines, rose1.filepath, "corrected-lines-")
    save_distribution_plot(distrib_walls_angular, rose1.filepath, 'angular-')
    save_distribution_plot(distrib_walls_angular_corrected_lines, rose1.filepath, "angular-corrected-lines-")
    #calculate emd
    #emd1 = EMD(distrib_walls_lines, distrib_walls_corrected_lines)
    u = list(distrib_walls_lines.keys())
    v = list(distrib_walls_corrected_lines.keys())
    u_weights = list(distrib_walls_lines.values())
    v_weights = list(distrib_walls_corrected_lines.values())
    emd = wasserstein_distance(u, v, u_weights, v_weights)
    """"
    u_angular = list(distrib_walls_angular.keys())
    v_angular = list(distrib_walls_angular_corrected_lines.keys())
    u_angular_weights = list(distrib_walls_angular.values())
    v_angular_weights = list(distrib_walls_angular_corrected_lines.values())
    emd_angulars = wasserstein_distance(u_angular, v_angular, u_angular_weights, v_angular_weights)
    """
    emd_angular = EMD_angular(distrib_walls_angular, distrib_walls_angular_corrected_lines)
    print("EMD: {}".format(emd))
    print("EMD walls directions: {}".format(emd_angular))
if __name__ == '__main__':
    main()