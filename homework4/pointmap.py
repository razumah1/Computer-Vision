import numpy as np

class PointMap:
    def __init__(self):
        self.array = np.array([0, 0, 0])

    def collect_points(self, tripoints):
        if len(tripoints) > 0:
            x_points = tripoints[0]
            y_points = -tripoints[1]  # Inverting y-axis
            z_points = -tripoints[2]  # Inverting z-axis

            array_to_project = np.column_stack((x_points, y_points, z_points))

            return array_to_project
