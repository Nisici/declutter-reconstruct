import runMe
from util.feature_correction import manhattan_directions
from util.feature_correction import correct_lines
from util.feature_correction import lines_directions
from util.disegna import draw_extended_lines, draw_walls
from util.feature_matching import lines_matching, average_distance_between_lines, distance_between_directions
from util.map_evaluation import avg_distance_walls_lines
import numpy as np
import argparse

def type_of_correction():
    print("0: Map exploration correction")
    print("1: Lines correction given a map")
    print("2: Manhattan correction")
    inp = int(input("Insert type of correction: "))
    return inp

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
if __name__ == '__main__':
    corr = type_of_correction()
    # Map exploration correction
    if corr == 0:
        rose1 = runMe.main()
        # take the main directions
        main_dirs1 = rose1.param_obj.comp[0], rose1.param_obj.comp[2]
        rose2 = runMe.main()
        corrected_lines = correct_lines(rose2.extended_segments_th1_merged[:-4], main_dirs1)
        draw_extended_lines(corrected_lines, rose2.walls, "corrected_lines", rose2.size, filepath=rose2.filepath)

        rose_gt = runMe.main()
        gt_dirs = rose_gt.param_obj.comp[0], rose_gt.param_obj.comp[2]
        lines_gt = rose_gt.extended_segments_th1_merged[:-4]
        print("Comparison between explored map and groundtruth.")
        #d1, d2 = distance_between_directions(main_dirs1, gt_dirs)
        print("Main dirs lines: {}".format(main_dirs1))
        print("Main dirs gt: {}".format(gt_dirs))
        """
        Compute distance between walls of map2 and corrected lines and confront it
        with the distance between walls of map2 and the lines of gt
        """


    #Correct map2 using directions of map1
    elif corr == 1:
        rose1 = runMe.main()
        main_dirs1 = rose1.param_obj.comp[0], rose1.param_obj.comp[2]
        rose2 = runMe.main()
        main_dirs2 = rose2.param_obj.comp[0], rose2.param_obj.comp[2]
        initial_lines = rose2.extended_segments_th1_merged[:-4]
        corrected_lines = correct_lines(initial_lines, main_dirs1)
        draw_extended_lines(corrected_lines, rose2.walls, "corrected_lines", rose2.size, filepath=rose2.filepath)
        print("Comparison between initial lines and corrected lines.")
        print("Main dirs first map: {}".format(main_dirs1))
        print("Main dirs second map: {}".format(main_dirs2))
        matchings_sum_distances = lines_matching(initial_lines,corrected_lines)
        avg_dist = matchings_sum_distances/len(corrected_lines)
        print("Average distance between lines matchings: {}".format(avg_dist))

    # Manhattan correction
    else:
        params = get_params()
        rose = runMe.main(filter_level=params.filter, sogliaClust=params.cluster)
        #take the main directions
        main_dirs = rose.param_obj.comp[0], rose.param_obj.comp[2]
        #correcte the main directions
        manhattan_dirs = manhattan_directions(main_dirs)
        #correct the lines
        corrected_lines = correct_lines(rose.extended_segments[:-4], manhattan_dirs)
        awd = avg_distance_walls_lines(rose.walls, corrected_lines, rose.original_binary_map, manhattan_dirs)
        #correct the walls
        corrected_walls = correct_lines(rose.walls, manhattan_dirs)
        draw_extended_lines(corrected_lines, rose.walls, "corrected_lines", rose.size, filepath=rose.filepath)
        draw_walls(rose.walls, "corrected_walls", rose.size, filepath=rose.filepath)
        draw_walls(rose.walls_projections, "walls_projections", rose.size, filepath=rose.filepath)
        acwd = avg_distance_walls_lines(corrected_walls, corrected_lines, rose.original_binary_map, manhattan_dirs)
        apwd = avg_distance_walls_lines(rose.walls_projections, corrected_lines, rose.original_binary_map, manhattan_dirs)
        print("Avg walls distance: {}".format(awd))
        print("Avg corrected walls_ang distance: {}".format(acwd))


