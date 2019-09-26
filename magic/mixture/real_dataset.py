import os
import copy

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import transforms3d as tf3d

from magic.shape_inference.utils import angle_to_quat

def axis_from_marker_pose(marker):
	# marker pose in magic frame
	marker_xyz = marker[:3,-1]

	# point along normal vector
	marker_normal_point = np.matmul(marker, np.array([0,0,1,1]).reshape(-1,1))
	marker_normal_point = marker_normal_point[:3].reshape(-1)

	# normal vector in magic frame
	marker_normal_vector = marker_normal_point - marker_xyz

	# axangle using arctan, forcing to be positive, as in data generation.
	axangle = np.arctan2(marker_normal_vector[1], marker_normal_vector[0])
	if axangle < 0:
		axangle = axangle + 2*np.pi

	# compute corresponding quaternion
	quat = angle_to_quat(axangle)

	# note: not using the quaternion from the AR tag.
	# using the normal computation above.
	# quat = tf3d.quaternions.mat2quat(tf_markers[i,:3,:3])
	axis = np.append(marker_xyz,quat)
	axis = torch.from_numpy(axis)
	return axis

class RealDataset(Dataset):
	def __init__(self, dir, n_dof, config, bounds, keep_columns, one_columns, full_labels, mask_val, obj):
		self.dirs = dir
		self.n_dof = n_dof
		self.config = torch.tensor(config)
		self.bounds = bounds
		self.keep_columns = keep_columns
		self.one_columns = one_columns
		self.full_labels = full_labels
		self.mask_val = mask_val
		self.obj = obj
		T_magic_to_cam = np.array([ [0. ,-1. , 0. , 0. ],
									[0. , 0. ,-1. , 0. ],
									[1. , 0. , 0. , 0. ],
									[0. , 0. , 0. , 1.0]])
		self.cam_to_magic = np.linalg.inv(T_magic_to_cam)
		self.images, self.axis, self.configs, self.og_depths = self.load(self.dirs)
		self.N = len(self.images)


	def load(self, dirs):

		# iterate over list and grab the ones which exist.
		out_images = []
		out_axes = []
		out_configs = []
		out_axangles = []
		og_depths = []
		for dir in dirs:

			print(dir)

			# load the marker poses
			marker_path = os.path.join(dir, 'marker_transforms.npy')
			markers = np.load(marker_path)

			# convert marker poses to magic frame
			if self.obj == 'cabinet2':
				tf_markers_left = np.matmul(self.cam_to_magic, markers[:,1])
				tf_markers_right = np.matmul(self.cam_to_magic, markers[:,0])
				print(tf_markers_left.shape)
				print(tf_markers_right.shape)
			else:
				tf_markers = np.matmul(self.cam_to_magic, markers)

			# get list of files
			segm_dir = os.path.join(dir, 'segmented')
			keep_indices = np.load(os.path.join(segm_dir, 'keep_indices.npy'))
			files = os.listdir(segm_dir)
			keep_files = [f for f in files if f.endswith('.npy') and not f.startswith('.')]
			sorted_file_list = sorted(keep_files)


			for i in range(len(markers)):
				fname = os.path.join(segm_dir, 'depth'+str(i).zfill(5)+'.npy')

				# skip if we've identified a reject
				if i not in keep_indices:
					continue

				try:

					# load depth image
					depth = np.load(fname)
					og_depth = copy.deepcopy(depth)

					# resize it for network input
					depth = cv2.resize(depth, (192,108))

					# cut out elements greater than 2m
					mask = depth < self.mask_val
					depth = depth * mask

					# normalize it (it was already converted to meters in segment_local_eyore or segment_from_json)
					depth = depth / 12.0

					# turn into tensor and concatenate
					depth = torch.tensor(depth).unsqueeze(0)
					depth = torch.cat((depth,depth,depth))

					# TODO: 2DoFs in this section. UGH
					# computing that fucking axis angle myself thanks to unreliable AR tags.

					if self.obj == 'cabinet2':
						axisL = axis_from_marker_pose(tf_markers_left[i])
						axisR = axis_from_marker_pose(tf_markers_right[i])
						axis = torch.stack((axisL, axisR))
					else:
						axis = axis_from_marker_pose(tf_markers[i])

					# append
					out_images.append(depth)
					og_depths.append(torch.tensor(og_depth))
					out_axes.append(axis.view(self.n_dof, 7))
					out_configs.append(self.config.view(self.n_dof,1))

					# print(marker_xyz)
					# print(marker_normal_point)
					# print(marker_normal_vector)
					# print(axangle * 180 / 3.14159)
					# print(quat)

				except Exception as e:
					# print(e)
					raise e

		out_images = torch.stack(out_images).float()
		og_depths = torch.stack(og_depths).float()
		out_axes = torch.stack(out_axes).float()
		out_configs = torch.stack(out_configs).float()

		if self.obj == 'refrigerator':
			out_axes = torch.cat((out_axes, out_axes), dim=1)
			out_configs = torch.cat((out_configs, out_configs), dim=1)

		print('images:', out_images.shape)
		print('axis labels:', out_axes.shape)
		print('config labels:', out_configs.shape)

		return out_images, out_axes, out_configs, og_depths

	def __len__(self):
		return self.N

	def __getitem__(self, idx):
		return {'depth': self.images[idx],
				'axis': self.axis[idx],
				'config': self.configs[idx],
				'label': torch.cat((self.axis[idx], self.configs[idx]), dim=1),
				'og_depth': self.og_depths[idx]}

# %%
# data = RealDataset('/Volumes/Passport/jun30-data/microwave0/', 1, 0.0, [], [], [])
#
# print('outside')
# datadict = data[0]
# print(datadict['depth'].shape)
# print(datadict['axis'].shape)
# print(datadict['config'].shape)
# print(datadict['label'].shape)
