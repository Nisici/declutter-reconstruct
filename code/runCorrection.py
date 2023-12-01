import runMe
from util.feature_correction import manhattan_directions
from util.feature_correction import manhattan_lines
from util.feature_correction import lines_directions
from util.disegna import draw_extended_lines
import numpy as np


if __name__ == '__main__':
    rose = runMe.main()
    #take the main directions
    main_dirs = rose.param_obj.comp[0], rose.param_obj.comp[2]
    #correcte the main directions
    manhattan_dirs = manhattan_directions(main_dirs)
    #correct the lines
    corrected_lines = manhattan_lines(rose.extended_segments[:-4], manhattan_dirs)
    draw_extended_lines(corrected_lines, rose.walls, "corrected_lines", rose.size, filepath=rose.filepath)
    print(np.array(lines_directions(corrected_lines)))