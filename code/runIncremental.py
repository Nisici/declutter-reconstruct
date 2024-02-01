import glob
import os
import shutil
import datetime
import tabulate
import numpy as np

import parameters as par
import minibatch

import FFT_MQ as fft
from matplotlib import pyplot as plt
from util.map_evaluation import avg_distance_walls_lines
from util.map_evaluation import distance_heat_map, pixel_to_wall

import re
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def check_int(name_folder):
    maps = sorted(os.listdir(name_folder))
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

"""
x axis: time-step
y axis: difference between 90° (being manhattan) and angle between two directions |d1-d2|
"""
def plot_incremental_dirs_manhattan(dirs, filepath):
    x = np.arange(0, len(dirs))
    manhattan = np.pi/2
    y = []
    for dir in dirs:
        print("dir1: {}, dir2: {}".format(dir[0], dir[2]))
        dir[0] = abs(dir[0])
        dir[2] = abs(dir[2])
        angle = abs(dir[0] - dir[2])
        deg1 = dir[0] * 180 / np.pi
        deg2 = dir[2] * 180 / np.pi
        angle_deg = angle * 180 / np.pi
        manhat_diff = abs(manhattan - angle) * 180 / np.pi
        y.append(manhat_diff)
        print("dir1: {}, dir2: {}".format(deg1, deg2))
        print("angle between: {}".format(angle_deg))
        print("angle diff manhat: {}".format(manhat_diff))
    plt.figure(figsize=(8,6))
    print(y)
    plt.plot(x, y)
    plt.savefig(filepath + "/" + "directions.png")

def plot_incremental_walls_distances(distances, filepath):
    x = np.arange(0, len(distances))
    plt.figure(figsize=(8,6))
    plt.plot(x, distances)
    plt.savefig(filepath + "/" + "wall_distances.png")

def main():
    # ----------------PARAMETERS OBJECTS------------------------
    # loading parameters from parameters.py
    parameters_object = par.ParameterObj()

    # loading path object with all path and name interesting for code
    paths = par.PathObj()

    # ----------------------------------------------------------

    # taking all the folders inside the path INPUT/IMGs
    list_dir = './data/INPUT/IMGs'
    map_folder = check_int(list_dir)
    t = list_dir + '/' + map_folder
    # asking the user what folder want to use
    map_name = check_int(t)
    paths.name_folder_input = map_name
    paths.path_folder_input = t + '/' + paths.name_folder_input
    if paths.name_folder_input == 'Bormann' or paths.name_folder_input == 'Bormann_furnitures':
        parameters_object.bormann = True
    else:
        parameters_object.bormann = False

    # saving the output folder where the output is saved
    make_folder('data/OUTPUT', map_folder)
    make_folder(os.path.join('data/OUTPUT/', map_folder), map_name)
    paths.path_folder_output = os.path.join('./data/OUTPUT/', map_folder, map_name)

    # apply rose to every map in directory
    directions = []
    distances = []

    for root, dirs, files in os.walk(paths.path_folder_input):
        for file in sorted(files, key=natural_sort_key):
            paths.metric_map_name = file
            paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
            if not paths.metric_map_name.endswith('.png') and not paths.metric_map_name.endswith('.jpg') and not paths.metric_map_name.endswith(
                    '.jpeg'):
                print('NOT AN IMAGE')
            else:
                rose = start_main(parameters_object, paths)
                directions.append(rose.param_obj.comp)
                avg_dist = avg_distance_walls_lines(rose.walls, rose.extended_segments_th1_merged, rose.original_binary_map, rose.param_obj.comp)
                distance_heat_map(rose.extended_segments, rose.walls,  paths.filepath , rose.original_map, rose.original_binary_map, rose.param_obj.comp)
                distances.append(avg_dist)
    #save directions in file
    with open(paths.path_folder_output + "/" + "directions.txt", 'w') as file:
        for dirs in directions:
            file.write("dirs: {}\n".format(dirs))
        file.write("\n")
        for dist in distances:
            file.write("wall dist: {}\n".format(dist))
    plot_incremental_dirs_manhattan(directions, paths.path_folder_output)
    plot_incremental_walls_distances(distances, paths.path_folder_output)

def start_main(parameters_object, paths):

    # -------------------EXECUTION---------------------------------------
    # ----------------------------------------------------------------
    # making the log folder
    paths.path_log_folder = os.path.join(paths.path_folder_output, paths.metric_map_name[:-4])
    running_time = str(datetime.datetime.now())[:-7].replace(' ', '@')
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
    return m

    # -------------------------------ENDING EXECUTION AND EVALUATION TIME------------------------------------



if __name__ == '__main__':
    main()
