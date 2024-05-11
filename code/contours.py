import runMe
import util.layout
import util.disegna as dsg
from util.map_evaluation import area_of_vertices
import cv2
from matplotlib import pyplot as plt

#area of contours as the sum of the areas of each contour
def contours_area(contours):
    area = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.0002 * perimeter, True)
        screen_cnt = approx
        vertices = []
        for c in screen_cnt:
            vertices.append([float(c[0][0]), float(c[0][1])])
        area += area_of_vertices(vertices)
    return area

if __name__ == '__main__':
    rose = runMe.main()
    original = rose.original_map.copy()
    decluttered = rose.orebro_img.copy()
    screen_cnt_or, vertices_or, contours_original = util.layout.external_contour(original)
    screen_cnt_dec, vertices_dec, contours_decluttered = util.layout.external_contour(decluttered)
    print("Contours original: {}".format(len(contours_original)))
    print("Contours decluttered: {}".format(len(contours_decluttered)))
    area_original = contours_area(contours_original)
    area_decluttered = contours_area(contours_decluttered)
    print("Area original: {}".format(area_original))
    print("Area decluttered: {}".format(area_decluttered))
