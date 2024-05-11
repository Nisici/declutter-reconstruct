import runMe
from util.map_evaluation import avg_distance_walls_lines
from util.feature_correction import correct_lines
from util.feature_matching import lines_matching
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.feature_correction import manhattan_directions
from util.disegna import draw_extended_lines
from util.map_evaluation import jaccard_idx, map_metric, distance_heat_map
import os

if __name__ == '__main__':
    rose = runMe.main()
    dirs = rose.param_obj.comp[0], rose.param_obj.comp[2]
    #jaccard_idx = jaccard_idx(rose.original_map, rose.orebro_img)
    #print("Jaccard val: {}".format(jaccard_idx))
    metric = map_metric(rose)
    distance_heat_map(rose.extended_segments, rose.walls, rose.filepath, rose.original_map, rose.original_binary_map, dirs)
    print("Metric val: {}".format(metric))
