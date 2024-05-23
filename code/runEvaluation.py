import runMe
from util.map_evaluation import avg_distance_walls_lines
from util.feature_correction import correct_lines
from util.feature_matching import lines_matching
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.feature_correction import manhattan_directions, correct_lines
from util.disegna import draw_extended_lines
from util.map_evaluation import jaccard_idx, map_metric, distance_heat_map
import os
import argparse
import parameters as par
import datetime
import shutil
import FFT_MQ as fft
import minibatch
from matplotlib.ticker import MaxNLocator

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter",
        type=float,
        default=0.18,
        help="Amount of clutter to remove",
    )
    parser.add_argument(
        "--cluster",
        type=float,
        default=5,
        help="sogliaLateraleClusterMura, più è alto meno cluster ci sono",
    )
    return parser.parse_args()

"""
x axis: time-step
y axis: difference between 90° (being manhattan) and angle between two directions |d1-d2| in degrees
"""
def plot_incremental_dirs_manhattan(dirs, filepath):
    x = np.arange(0, len(dirs))
    manhattan = np.pi/2
    y = []
    for dir in dirs:
        angle = abs(dir[0] - dir[2])
        manhat_diff = abs(manhattan - angle) * 180 / np.pi
        y.append(manhat_diff)
    plt.figure(figsize=(8,6))
    plt.plot(x, y)
    plt.xticks(range(min(x), max(x) + 1))
    plt.savefig(os.path.join(filepath, "directions.png"))

def plot_incremental_walls_distances(distances, filepath, name="wall_distances"):
    x = np.arange(0, len(distances))
    plt.figure(figsize=(8,6))
    plt.plot(x, distances)
    plt.xticks(range(min(x), max(x) + 1))
    plt.savefig(os.path.join(filepath, name))

def plot_incremental_metric(metrics, filepath):
    x = np.arange(0, len(metrics))
    plt.figure(figsize=(8,6))
    plt.plot(x, metrics)
    plt.xticks(range(min(x), max(x) + 1))
    plt.savefig(os.path.join(filepath, "metrics.png"))


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
    make_folder(paths.path_folder_output, "plots-stats")

    # apply rose to every map in directory
    directions = []
    distances = []
    distances_manhattan = []
    metrics = []

    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    for root, dirs, files in os.walk(paths.path_folder_input):
        for file in sorted(files, key=natural_sort_key):
            paths.metric_map_name = file
            paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
            if not paths.metric_map_name.endswith('.png') and not paths.metric_map_name.endswith('.jpg') and not paths.metric_map_name.endswith(
                    '.jpeg'):
                print('NOT AN IMAGE')
            else:
                params = get_params()
                parameters_object.filter_level, parameters_object.sogliaLateraleClusterMura = params.filter, params.cluster
                try:
                    rose = start_main(parameters_object, paths)
                except Exception:
                    print("cant find directions")
                    continue
                dirs = [rose.param_obj.comp[0], rose.param_obj.comp[2]]
                manhattan_dirs = manhattan_directions(dirs)
                manhattan_lines = correct_lines(rose.extended_segments, manhattan_dirs)[:-4]
                draw_extended_lines(manhattan_lines, rose.walls, "corrected_lines", rose.size, filepath=rose.filepath)
                directions.append(rose.param_obj.comp)
                avg_dist_walls_lines = avg_distance_walls_lines(rose.walls, rose.extended_segments, rose.original_binary_map, dirs)
                avg_dist_walls_manhattan_lines = avg_distance_walls_lines(rose.walls, manhattan_lines,
                                                                          rose.original_binary_map, dirs,
                                                                          corrected_lines=True)
                metric = map_metric(rose)
                distance_heat_map(rose.extended_segments, rose.walls,  rose.filepath , rose.original_map, rose.original_binary_map, dirs)
                distances.append(avg_dist_walls_lines)
                distances_manhattan.append(avg_dist_walls_manhattan_lines)
                metrics.append(metric)
                degrees = abs(dirs[0] - dirs[1]) * 180 / np.pi
                with open(os.path.join(rose.filepath, "evaluation.txt"), 'w') as file:
                    file.write("Degrees: {}\n".format(degrees))
                    file.write("Avg dist wall lines: {}\n".format(avg_dist_walls_lines))
                    file.write("Avg dist wall manhattan lines: {}\n".format(avg_dist_walls_manhattan_lines))
                    file.write("Metric val: {}".format(metric))

                print("Degrees: {}".format(degrees))
                print("Avg dist wall lines: {}".format(avg_dist_walls_lines))
                print("Avg dist wall manhattan lines: {}".format(avg_dist_walls_manhattan_lines))
                print("Metric val: {}".format(metric))
    time = str(datetime.datetime.now())
    plots_path = os.path.join(paths.path_folder_output, "plots-stats")
    make_folder(plots_path, time)
    plots_path = os.path.join(plots_path, time)
    with open(os.path.join(plots_path, "params.txt"), 'w') as file:
        file.write("filter_level: {}\n".format(parameters_object.filter_level))
        file.write("sogliaLateraleClusterMura: {}\n".format(parameters_object.sogliaLateraleClusterMura))
    with open(os.path.join(plots_path, "stats.txt"), 'w') as file:
        meanD = np.mean(directions)
        varianceD = np.std(directions)
        meanWL = np.mean(distances)
        varianceWL = np.std(distances)
        meanWM = np.mean(distances_manhattan)
        varianceWM = np.std(distances_manhattan)
        meanM = np.mean(metrics)
        varianceM = np.std(metrics)
        file.write("Degree diff manhattan mean: {}\n".format(meanD))
        file.write("Degrees diff manhattan variance: {}\n".format(varianceD))
        file.write("Avg dist wall lines mean: {}\n".format(meanWL))
        file.write("Avg dist wall lines variance: {}\n".format(varianceWL))
        file.write("Avg dist wall manhattan lines mean: {}\n".format(meanWM))
        file.write("Avg dist wall manhattan lines variance: {}\n".format(varianceWM))
        file.write("Metric mean: {}\n".format(meanM))
        file.write("Metric variance: {}\n".format(varianceM))

    plot_incremental_dirs_manhattan(directions, plots_path)
    plot_incremental_walls_distances(distances, plots_path)
    plot_incremental_walls_distances(distances_manhattan, plots_path, "wall_distances_manhattan")
    plot_incremental_metric(metrics, plots_path)

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

if __name__ == '__main__':
    main()
