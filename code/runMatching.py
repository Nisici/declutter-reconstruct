import glob
import os
import shutil
import datetime
import tabulate
import numpy as np

import parameters as par
import minibatch

import FFT_MQ as fft
from util.feature_matching import lines_matching
from util.feature_matching import average_distance_between_lines
from util.feature_matching import average_distance_between_points
from util.feature_matching import contour_matching
from object.Segment import radiant_inclination
from util.disegna import draw_extended_lines
from object.Segment import segments_distance
import PIL.Image

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

def find_main_lines(lines, num_lines=2):
    angles_with_lines = {}
    for l in lines:
        angle = radiant_inclination(l.x1, l.y1, l.x2, l.y2)
        angle = angle % (np.pi)  # Ensure the angle is within [0, 2*pi)
        angle = angle * 180/np.pi
        if angle not in angles_with_lines.keys():
            angles_with_lines[angle] = [l]
        else:
            angles_with_lines[angle].append(l)
    main_lines = []
    #for each angle find the main line
    for k in angles_with_lines.keys():
        main_lines.append(max(angles_with_lines[k], key=lambda m: m.weight))
    #find the n best lines where n is the number of main directions find with rose
    main_lines = sorted(main_lines, key=lambda m:m.weight)[-num_lines:]
    return main_lines

#finds the main lines of rose1 and rose2 and then compares them printing their directions.
def compare_main_lines(rose1, par1, rose2, par2):
    num_main_dir1 = int(len(par1.comp) / 2)
    main_lines_1 = find_main_lines(rose1.extended_segments, num_main_dir1)
    draw_extended_lines(main_lines_1, rose1.walls, "main_lines", rose1.size, filepath=rose1.filepath)
    num_main_dir2 = int(len(par2.comp) / 2)
    main_lines_2 = find_main_lines(rose2.extended_segments, num_lines=num_main_dir2)
    draw_extended_lines(main_lines_2, rose2.walls, "main_lines", rose2.size, filepath=rose2.filepath)
    dir11 = radiant_inclination(main_lines_1[0].x1, main_lines_1[0].y1,main_lines_1[0].x2,main_lines_1[0].y2)%np.pi
    dir21 = radiant_inclination(main_lines_1[1].x1, main_lines_1[1].y1, main_lines_1[1].x2, main_lines_1[1].y2)%np.pi
    dir12 = radiant_inclination(main_lines_2[0].x1, main_lines_2[0].y1, main_lines_2[0].x2, main_lines_2[0].y2)%np.pi
    dir22 = radiant_inclination(main_lines_2[1].x1, main_lines_2[1].y1, main_lines_2[1].x2, main_lines_2[1].y2)%np.pi
    print("Main dir1: {}, main dir2: {}".format(par1.comp[0]%np.pi, par1.comp[2]%np.pi))
    print("Dir1: {}, Dir2: {}".format(dir11, dir21))
    print("Main dir1: {}, main dir2: {}".format(par2.comp[0]%np.pi, par2.comp[2]%np.pi))
    print("Dir1: {}, Dir2: {}".format(dir12, dir22))


def matching_all_lines(rose1, rose2, dirs1, dirs2):
    lines1 = rose1.extended_segments[:-4] #remove contour lines
    lines2 = rose2.extended_segments[:-4]
    lines_image_1 = PIL.Image.open(rose1.filepath + "/Extended_Lines/7a_extended_lines.png")
    lines_image_2 = PIL.Image.open(rose2.filepath + "/Extended_Lines/7a_extended_lines.png")
    lines_image_1 = np.array(lines_image_1.convert('RGB'))
    lines_image_2 = np.array(lines_image_2.convert('RGB'))
    avg_dist_matchings = lines_matching(lines1, lines2, dirs1, dirs2)
    print("Average distance between lines matchings: {}".format(avg_dist_matchings))


if __name__ == '__main__':
    m1, par1 = rose_single_action()
    m2, par2 = rose_single_action()
    #dirs[0] = horizontal direction, dirs[1] = vertical direction
    dirs1 = par1.comp[0], par1.comp[2]
    dirs2 = par2.comp[0], par2.comp[2]
    tol = 1
    if abs(dirs2[0]-dirs1[0]) >= tol:
        t = dirs2[0]
        dirs2[0] = dirs2[1]
        dirs2[1] = t
    matching_all_lines(m1, m2, dirs1, dirs2)
    #matching_contour(m1, m2)
