import numpy as np
import time
import pptk

# import pcl
from tqdm import tqdm
# from reconstruction.icp import icp
import reconstruction.calibrations as calib

class Reconstructor(object):
    """ Make pointcloud from one/many depth image(s).

    reconstructed pointcloud lives in self.points
    add_cloud to add to self.points; optionally register and filter
    segment method will segment a depth image using min and max pt.
    """

    def __init__(self,
                 projection_matrix,
                 min_pt=[0.8, -0.343, 0.534],
                 max_pt=[1.797,  0.6,  1.067],
                 invalid_val = 99):
        super(Reconstructor, self).__init__()
        self.fx = projection_matrix[0,0]
        self.fy = projection_matrix[1,1]
        self.cx = projection_matrix[0,2]
        self.cy = projection_matrix[1,2]
        self.max_depth = 11.9 # max depth in METERS, eventually a parameter
        self.points = np.zeros((1,3), dtype=np.float32)
        self.min_pt = min_pt
        self.max_pt = max_pt
        self.icp_transforms=[]
        self.invalid_val = invalid_val

    def point_cloud(self, depth):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths in meters. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

        valid = (depth > 0) & (depth < self.max_depth)
        z = np.where(valid, depth, self.invalid_val)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)
        return np.dstack((x, y, z)).astype(np.float32)

    def add_cloud(self, depth, camera_pose, register=False, filter=False, downsample=False):
        """ Add points from a depth image to self.points,
            optionally filter and register using PCA
        """
        # create pcd from depth image
        pcd = self.point_cloud(depth)
        points = pcd.reshape(-1,3)

        filtered_points=points[~np.isnan(points).any(axis=1)]

        # transform into map frame
        transformed_pcd = self.transform(filtered_points, camera_pose)

        # filter?
        if filter == True:

            # box filter
            transformed_pcd = self.box_filter(transformed_pcd,
                                              self.min_pt,
                                              self.max_pt)

        # register?
        if register == True:
            icp_transf, score = icp(transformed_pcd.astype(np.float32), self.points.astype(np.float32))
            transformed_pcd = self.transform(transformed_pcd, icp_transf)
            self.icp_transforms.append(icp_transf)

        else:
            score=0

        # add the points if the icp error is low enough
        if score < 1e8:

            old_n_pts = len(self.points)
            self.points = np.vstack((self.points,transformed_pcd))

            if downsample:# # downsample with voxel for efficiency
                cloud = pcl.PointCloud()
                cloud.from_array(self.points.astype(np.float32))
                sor = cloud.make_voxel_grid_filter()
                sor.set_leaf_size(0.01, 0.01, 0.01)
                cloud_filtered = sor.filter()
                self.points = cloud_filtered.to_array()

            new_n_pts = len(self.points)
            return new_n_pts - old_n_pts

        else:

            print('ICP score too high, not adding points.')
            return 0


    def transform(self, points, camera_pose):
        """ Converts a list of points (-1,3) to world frame using camera pose """
        points_T = self.homogenize(points)
        transformed_pts = np.matmul(camera_pose, points_T)
        return transformed_pts.transpose((1,0))[:,:3]

    def box_filter(self,points,min_point,max_point):
        """ Throws out all points not in axis-aligned box in world frame. """
        mask = (points > min_point) & (points < max_point)
        out = points[mask.all(axis=1)]
        return out

    def homogenize(self,points):
        """ Convert points to homogenous (4,-1) for transforms """
        n_pts = len(points)
        homog_points = np.hstack((points, np.ones((n_pts,1))))
        points_T = homog_points.transpose((1,0))
        return points_T

    def segment(self, image, camera_pose):
        """ Segments a depth image using a box defined by self.min_pt and self.max_pt in world frame"""

        # make a pointcloud
        pcd = self.point_cloud(image)
        points = pcd.reshape(-1,3)

        # transform into map frame
        transformed_pcd = self.transform(points, camera_pose)

        # reshape back into image shape
        pcd = transformed_pcd.reshape(image.shape[0], image.shape[1], 3)

        # now compute mask
        mask = (pcd > self.min_pt) & (pcd < self.max_pt)
        # print(mask)
        pixelwise_mask = mask.all(axis=2)
        return pixelwise_mask * image

def test():
    ## load, normalize depth
    # depth_image = np.load('/Users/abba/projects/magic/magic-reality/data/maskedfull/masked_full0000.npy')
    # depth_image=np.load('/Users/abba/projects/magic/magic-reality/data_collection/data/microwave/microwave-closed/depth00000.npy')
    # depth_image = depth_image / 1000.0 # convert to meters
    # print(depth_image.max())
    # print(depth_image.min())

    ## create reconstructor

    # these min and max are for data/floor4/fridge. feel free to experiment
    recon = Reconstructor(calib.color_cam_matrix,
                          min_pt = [-0.919, 0.42 , 0.2],
                          max_pt = [0 ,1.172, 1.778])

    ## create pointcloud
    # pcd = recon.point_cloud(depth_image)
    # points = pcd.reshape(-1,3)
    # filtered_points = np.array([p for p in points if not np.isnan(p[2])])
    # print(filtered_points.shape)
    # v=pptk.viewer(filtered_points)
    # _ = raw_input('press enter to continue')

    ## then, test transformed pointcloud
    # transforms = np.load('/Users/abba/projects/magic/magic-reality/data_collection/data/microwave/microwave-closed/transforms.npy')
    # print(transforms.shape)
    # tf_cloud = recon.transform(points, transforms[0])
    # v=pptk.viewer(tf_cloud)

    ## load data
    # dir = '/Users/abba/projects/magic/magic-reality/data_collection/data/microwave/microwave-closed/'
    dir = '/Users/abba/projects/magic/magic-reality/data_collection/data/floor4/fridge/'
    transforms = np.load(dir+'transforms.npy')

    # incrementally transform and register
    n_pts = [0]
    rgb = np.array([0,0,0])
    N = 10
    start=1
    for i in tqdm(np.arange(start,N,1)):
        x=np.load(dir+'depth'+str(i).zfill(5)+'.npy') / 1000.0

        n_new_pts = recon.add_cloud(x, transforms[i],
                                    filter=False,
                                    register= (i-start) > 0,
                                    downsample=True)

        new_color = np.random.rand(1,3)
        new_ones = np.ones((n_new_pts, 3))
        rgb = np.vstack((rgb, new_ones * new_color))

    v=pptk.viewer(recon.points, rgb)

    # np.save('merged_fridge_points.npy', recon.points)



if __name__ == '__main__':
    test()
