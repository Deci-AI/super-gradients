import unittest

import numpy as np

from super_gradients.training.samples import PoseEstimationSample


class PoseEstimationSampleTest(unittest.TestCase):
    def test_compute_joints_areas(self):
        joints = np.array(
            [
                [[10, 20, 1], [20, 30, 1], [15, 25, 1], [16, 23, 1]],
                [[10, 20, 1], [20, 30, 1], [15, 25, 0], [16, 23, 0]],
                [[15, 25, 0], [10, 20, 0], [20, 30, 1], [15, 25, 1]],
                [[15, 25, 0], [10, 20, 0], [20, 30, 1], [15, 25, 0]],
                [[15, 25, 0], [10, 20, 0], [20, 30, 0], [15, 25, 0]],
            ]
        )

        areas = PoseEstimationSample.compute_area_of_joints_bounding_box(joints)
        self.assertAlmostEquals(areas[0], 100)
        self.assertAlmostEquals(areas[1], 100)
        self.assertAlmostEquals(areas[2], 25)
        self.assertAlmostEquals(areas[3], 0)
        self.assertAlmostEquals(areas[4], 0)
