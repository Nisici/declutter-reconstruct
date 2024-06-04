import runMe
from util.map_evaluation import avg_distance_walls_lines
from util.feature_correction import correct_lines
from util.feature_matching import lines_matching
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.feature_correction import manhattan_directions, correct_lines
from util.disegna import draw_extended_lines, draw_walls
from util.map_evaluation import jaccard_idx, map_metric, distance_heat_map, map_metric_manhattan
import os
import argparse
import parameters as par
import datetime
import shutil
import FFT_MQ as fft
import minibatch
from matplotlib.ticker import MaxNLocator
import re
""""
Evaluate maps using rose, calculate: main directions, avg distance between walls and the extended lines, 
avg distance between walls and manhattan extended lines (90°), metric (not yet finished).
You can evaluate a single map passing the command line argument --single True. This will make a file evaluate.txt
with the stats, heatmap_on_original.png to see the distance between walls and lines colored on the map (blue = max dist,
red = min dist), corrected_lines.png shows how the extended lines have been corrected to be manhattan.
You can also evaluate a batch of maps from a folder with --single False (default). 
If you evaluate a batch of map for each of them you will produce the same result as for the single map case and a new folder
plot-stats which contains plots showing the values for each map, following alphabetical order, and mean and std for each val.

"""

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
    parser.add_argument(
        "--single",
        type=bool,
        default=False,
        help="Run eval only on one image",
    )
    return parser.parse_args()


def plot_incremental_vals(distances, filepath, name="plot", labels=None):
    x = labels
    if not labels:
        x = np.arange(0, len(distances))
        plt.xticks(range(min(x), max(x) + 1))
    plt.figure(figsize=(8,6))
    plt.plot(x, distances)
    plt.savefig(os.path.join(filepath, name))

#directions: list of angles
#returns: list of difference between angles and 90° in degrees
def dirs_diff_manhattan(dirs):
    manhattan = np.pi / 2
    y = []
    for dir in dirs:
        angle = abs(dir[0] - dir[1])
        manhat_diff = abs(manhattan - angle) * 180 / np.pi
        y.append(manhat_diff)
    return y

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def evaluate_single_map(paths, parameters_object):
    try:
        rose = start_main(parameters_object, paths)
    except Exception:
        raise Exception
    dirs = [rose.param_obj.comp[0], rose.param_obj.comp[2]]
    manhattan_dirs = manhattan_directions(dirs)
    manhattan_lines = correct_lines(rose.extended_segments, manhattan_dirs)
    corrected_walls = correct_lines(rose.walls, manhattan_dirs)
    draw_extended_lines(manhattan_lines, rose.walls, "corrected_lines", rose.size, filepath=rose.filepath)
    draw_walls(corrected_walls, "corrected_walls", rose.size, filepath=rose.filepath)
    avg_dist_walls_lines = avg_distance_walls_lines(rose.walls, rose.extended_segments, rose.original_binary_map, dirs)
    avg_dist_walls_manhattan_lines = avg_distance_walls_lines(rose.walls, manhattan_lines,
                                                              rose.original_binary_map, manhattan_dirs,)
    avg_corrected_walls_distance = avg_distance_walls_lines(corrected_walls, manhattan_lines, rose.original_binary_map, manhattan_dirs)
    metric = map_metric(rose)
    distance_heat_map(rose.extended_segments, rose.walls, rose.filepath, rose.original_map, rose.original_binary_map,
                      dirs)
    degrees = abs(dirs[0] - dirs[1]) * 180 / np.pi
    metric_manhattan = map_metric_manhattan(rose, manhattan_lines, manhattan_dirs)
    with open(os.path.join(rose.filepath, "evaluation.txt"), 'w') as file:
        file.write("Degrees: {}\n".format(degrees))
        file.write("Avg dist wall lines: {}\n".format(avg_dist_walls_lines))
        file.write("Avg dist wall manhattan lines: {}\n".format(avg_dist_walls_manhattan_lines))
        file.write("Metric val: {}\n".format(metric))
        file.write("Metric manhattan val: {}\n".format(metric_manhattan))
        file.write("Corrected walls distance: {}".format(avg_corrected_walls_distance))
    return dirs, avg_dist_walls_lines, avg_dist_walls_manhattan_lines, metric, metric_manhattan, avg_corrected_walls_distance

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

    params = get_params()
    parameters_object.filter_level, parameters_object.sogliaLateraleClusterMura = params.filter, params.cluster

    if params.single:
        make_folder('data/OUTPUT', 'SINGLEMAP')
        paths.path_folder_output = './data/OUTPUT/SINGLEMAP'
        # asking what map to use
        paths.metric_map_name = check_int(paths.path_folder_input)
        paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
        evaluate_single_map(paths, parameters_object)

    #APPLY ROSE TO EVERY MAP IN DIRECTORY
    else:
        directions = []
        distances = []
        distances_manhattan = []
        metrics = []
        metrics_manhattan = []
        corrected_walls_distances = []
        labels = [] # for plotting
        make_folder(paths.path_folder_output, "plots-stats")
        for root, dirs, files in os.walk(paths.path_folder_input):
            for file in sorted(files, key=natural_sort_key):
                paths.metric_map_name = file
                paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
                if not paths.metric_map_name.endswith('.png') and not paths.metric_map_name.endswith('.jpg') and not paths.metric_map_name.endswith(
                        '.jpeg'):
                    print('NOT AN IMAGE')
                else:
                    try:
                        direction, avg_dist_walls_lines, avg_dist_walls_manhattan_lines, metric, metric_manhattan, avg_corrected_walls_distance\
                            = evaluate_single_map(paths, parameters_object)
                    except Exception:
                        print("Can't find directions")
                        continue
                    directions.append(direction)
                    distances.append(avg_dist_walls_lines)
                    distances_manhattan.append(avg_dist_walls_manhattan_lines)
                    metrics.append(metric)
                    metrics_manhattan.append(metric_manhattan)
                    corrected_walls_distances.append(avg_corrected_walls_distance)
                    label = re.findall(r'\d+', file)[-1]
                    labels.append(label)
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
        dirs_diff_manh = dirs_diff_manhattan(directions)
        plot_incremental_vals(dirs_diff_manh, plots_path, "directions", labels)
        plot_incremental_vals(distances, plots_path, "wall_distances", labels)
        plot_incremental_vals(distances_manhattan, plots_path, "wall_distances_manhattan", labels)
        plot_incremental_vals(corrected_walls_distances, plots_path, "corrected_walls_distances", labels)
        #plot_incremental_vals(metrics, plots_path, "metrics", labels)
        #plot_incremental_vals(metrics_manhattan, plots_path, "metrics_awd_manhattan", labels)

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
