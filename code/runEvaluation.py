import runMe
from util.map_evaluation import avg_distance_walls_lines
from util.feature_correction import correct_lines
from util.feature_matching import lines_matching
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.feature_correction import manhattan_directions
from util.disegna import draw_extended_lines
from util.map_evaluation import similarity_gt
from util.map_evaluation import jaccard_idx
import os
def type_of_evaluation():
    print("0: Structural similarity between map and gt")
    print("1: Distance walls lines")
    print("2: Distance walls from lines of gt")
    print("3: Distance wslls with lines, corrected lines and gt ")
    inp = int(input("Insert type of evaluation: "))
    return inp

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
    eval = type_of_evaluation()
    if eval==0:
        list_dir = './data/INPUT/IMGs'
        t = list_dir + '/' + check_int(list_dir)
        name_folder_input = t + '/' + check_int(t)
        map_path = name_folder_input + '/' + check_int(name_folder_input)
        map = Image.open(map_path).convert('L')
        t = list_dir + '/' + check_int(list_dir)
        name_folder_input = t + '/' + check_int(t)
        gt_path = name_folder_input + '/' + check_int(name_folder_input)
        gt = Image.open(gt_path).convert('L')
        map = np.array(map)
        gt = np.array(gt)
        sim_score = similarity_gt(map, gt)
        jacc_score = jaccard_idx(map, gt)
        print("Similarity score: {}".format(sim_score))
        print("Jaccard index: {}".format(jacc_score))
    if eval != 0:
        rose = runMe.main()
        avg_dist = avg_distance_walls_lines(rose.walls, rose.extended_segments[:-4], rose.original_binary_map)
    elif eval == 1:
        rose_gt = runMe.main()
        print("Calculating average distance between walls of map and lines of gt.")
        dirs = rose_gt.param_obj.comp[0], rose_gt.param_obj.comp[2]
        avg_dist = avg_distance_walls_lines(rose.walls, rose_gt.extended_segments[:-4], rose.original_binary_map, dirs=dirs)

    else:
        rose_gt = runMe.main()
        corrected_lines = correct_lines(rose.extended_segments[:-4], (rose_gt.param_obj.comp[0], rose_gt.param_obj.comp[2]))
        draw_extended_lines(corrected_lines, rose.walls, "corrected", rose.size, filepath=rose.filepath)
        print("Calculating average distance between walls of map and corrected lines.")
        dirs = rose_gt.param_obj.comp[0], rose_gt.param_obj.comp[2]
        avg_dist_gt = avg_distance_walls_lines(rose.walls, rose_gt.extended_segments[:-4], rose.original_binary_map, dirs=dirs)
        avg_dist_corr = avg_distance_walls_lines(rose.walls, corrected_lines, rose.original_binary_map, dirs=dirs)
        avg_dist = avg_distance_walls_lines(rose.walls, rose.extended_segments[:-4], rose.original_binary_map)
        print("Avg distance walls, lines using gt: {}".format(avg_dist_gt))
        print("Avg distance walls, lines using corrected lines: {}".format(avg_dist_corr))
    print("Avg distance walls, lines using map: {}".format(avg_dist))
