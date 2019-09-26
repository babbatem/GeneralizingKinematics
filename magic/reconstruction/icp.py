# -*- coding: utf-8 -*-
# How to use iterative closest point
# http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

import pcl
import random
import numpy as np

# from pcl import icp, gicp, icp_nl

def icp(points_in, points_out):
    '''Does what it says it will
    Returns input points in

    '''
    cloud_in = pcl.PointCloud()
    cloud_out = pcl.PointCloud()

    cloud_in.from_array(points_in)
    cloud_out.from_array(points_out)

    icp = cloud_in.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)
    # print(converged)
    # print(transf)
    # print(estimate)
    # print(fitness)
    return transf, fitness
