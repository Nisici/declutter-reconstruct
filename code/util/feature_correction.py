import numpy as np
from object.Segment import radiant_inclination
from object.ExtendedSegment import ExtendedSegment
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


"""
lines: list of ExtendedSegment
directions: list of manhattan directions (floats)
returns -> list of ExtendedSegment that are manhattan between each other

Rotates the lines so that their inclination is equal to the closest direction in directions
"""
def manhattan_lines(lines, directions):
    new_lines = []
    for l in lines:
        line_dir = radiant_inclination(l.x1, l.y1, l.x2, l.y2)
        #directions = [0, np.pi/2]
        main_dir = min(directions, key=lambda x: abs(line_dir - x))
        rot_angle = main_dir - line_dir
        new_line = rotate_line(l, rot_angle)
        line_dir = radiant_inclination(new_line.x1, new_line.y1, new_line.x2, new_line.y2)
        new_lines.append(new_line)
    return new_lines

"""
line: ExtendedSegment
direction: float (radians)
Rotate the line by a certain angle
returns a new line rotated 
"""
def rotate_line(line, angle):
    cx = (line.x1 + line.x2) / 2
    cy = (line.y1 + line.y2) / 2
    # counterclockwise
    x1 = (line.x1 - cx) * np.cos(angle) - (line.y1 - cy) * np.sin(angle) + cx
    y1 = (line.x1 - cx) * np.sin(angle) + (line.y1 - cy) * np.cos(angle) + cy
    x2 = (line.x2 - cx) * np.cos(angle) - (line.y2 - cy) * np.sin(angle) + cx
    y2 = (line.x2 - cx) * np.sin(angle) + (line.y2 - cy) * np.cos(angle) + cy
    new_line = ExtendedSegment((x1, y1), (x2, y2), None, None)
    return new_line


"""
precondition: len(directions) = 2
directions: list of directions (floats) 
returns -> list of manhattan directions
"""
def manhattan_directions(directions):
    #minimization problem: find the minimum adjustment to the initial directions so that they become manhattan
    # Objective function to minimize: x1 + x2
    def objective_function(x):
        return x[0] + x[1]

    # Constraints functions
    # angle = abs(angle1 - angle2)
    # angle + displacement <= pi/2 + epsilon
    # 3.15 <= 3.14 + 0.01
    def lower_than(x, angle1, angle2, margin=0):
        return -abs(x[0] + angle1 - (x[1] + angle2)) + np.pi/2 + margin

    # angle = abs(dir1 - dir2)
    # angle + displacement >= pi/2 - epsilon
    # 3.13 >= 3.14 - 0.01
    def greater_than(x, angle1, angle2, margin=0):
        return abs(x[0] + angle1 - (x[1] + angle2)) - np.pi/2 + margin

    def equal_to(x, angle1, angle2):
        return abs(x[0] + angle1 - (x[1] + angle2)) - np.pi/2
    def positive_obj_fun(x):
        return x[0] + x[1]

    # Initial guess
    x0 = np.array([0.0, 0.0])
    epsilon = 0.01
    # Constraints
    constraints = ({'type': 'ineq', 'fun': greater_than, 'args': (directions[0], directions[1], epsilon)},
                {'type': 'ineq', 'fun': lower_than, 'args': (directions[0], directions[1], epsilon)},
                {'type': 'ineq', 'fun': positive_obj_fun}
    )
    # Optimization
    result = minimize(objective_function, x0, constraints=constraints)
    print("Optimization result: {}".format(result.success))
    displacement = result.x
    manhattan_dirs = directions + displacement
    print("Initial directions: {}".format(directions))
    print("Displacement: {}".format(displacement))
    print("Manhattan directions: {}".format(manhattan_dirs))
    return manhattan_dirs


"""
lines: list of ExtendedSegment
Returns the unique directions of the list of lines
"""
def lines_directions(lines):
    directions = [radiant_inclination(lines[0].x1, lines[0].y1, lines[0].x2, lines[0].y2)]
    tolerance = 0.001
    flag = True
    for l in lines:
        angle = radiant_inclination(l.x1, l.y1, l.x2, l.y2)
        for d in directions:
            if abs(angle - d) < tolerance:
                flag = False
                break
        if flag:
            directions.append(angle)
        flag = True
    return directions

