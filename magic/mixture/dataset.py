import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from magic.noise_models import Distractor

# %%
class MixtureDataset(Dataset):
	def __init__(self,
	    	     ntrain,
				 root_dir,
				 bounds=[],
				 n_dof=1,
				 normalize=False,
				 transform=None,
				 test=False,
				 keep_columns=[],
				 one_columns=[],
				 preserve_labels=False,
				 distractor_prob=0.0):
		super(MixtureDataset, self).__init__()
		'''
			shapes and kinematics and stuff
		'''
		self.root_dir = root_dir
		self.labels_frame = pd.read_csv(os.path.join(root_dir, 'params.csv'), header=None)
		self.raw_labels = self.labels_frame.values
		self.length=ntrain
		self.n_dof = n_dof
		self.transform=transform

		if not test:
			if normalize:
				if bounds == []:
					self.compute_bounds()
					np.save(os.path.join(root_dir, 'bounds.npy'), self.bounds)
				else:
					self.bounds = bounds
				print('normalizing.')
				self.normalize_labels()
			else:
				self.bounds=[]

		self.raw_torch = torch.from_numpy(self.raw_labels).float()
		axes=self.get_axes()
		radii = self.get_radii()
		configs=self.get_angles()
		geoms = self.get_geometry()
		poses = self.get_poses()

		# concatenate all the quantities into one labels datastructure
		self.full_labels = torch.cat((axes,
							configs,
							radii,
							geoms,
							poses), dim=1)

		if keep_columns == []:
			# compute which columns are constant zeros or constant ones
			x = self.full_labels
			self.zero_columns = [i for i in range(x.shape[1]) if not x[:,i].byte().any()]
			self.one_columns = [i for i in range(x.shape[1]) if x[:,i].byte().all()]
			self.keep_columns = [i for i in range(x.shape[1]) if (i not in self.zero_columns) and (i not in self.one_columns)]
		else:
			self.keep_columns=keep_columns
			self.one_columns=one_columns

		# throw out constant columns for stability!
		print('full labels shape', self.full_labels.shape)
		self.labels = self.full_labels[:, self.keep_columns]
		print('labels shape', self.labels.shape)

		# set up distractor in the background
		self.distractor_prob = distractor_prob
		self.distractor = Distractor(trans_weight = 0.0,
									 rotate = False)

		print(self.keep_columns)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):

		# load depth
		path=os.path.join(self.root_dir, 'depth'+str(idx).zfill(6)+'.pt')
		depth = torch.load(path)

		# random other depth image
		if np.random.rand() < self.distractor_prob:
			other_idx = np.random.randint(self.length)
			path2=os.path.join(self.root_dir, 'depth'+str(other_idx).zfill(6)+'.pt')
			depth2 = torch.load(path2)
			depth = self.distractor(depth,depth2)

		depth = depth.unsqueeze(0).float()

		if self.transform is not None:
			depth = self.transform(depth)

		label = self.labels[idx]
		depth=torch.cat((depth,depth,depth))
		sample = {'depth': depth,
				  'label': label}
		return sample

	def get_axes(self):
		# print('Loading axes.')
		if self.n_dof == 1:
			axes=self.raw_torch.narrow(1,5,7)
		else:
			axes = self.raw_torch.narrow(1,5,14)
		# print(axes.size())
		return axes

	def get_radii(self):
		# print('Loading radii.')
		if self.n_dof == 1:
			radii = self.raw_torch.narrow(1,12,3)
		else:
			radii = self.raw_torch.narrow(1,19,6)
		# print(radii.size())
		return radii

	def get_geometry(self):
		geom = self.raw_torch.narrow(1,1,4)
		return geom

	def get_poses(self):
		if self.n_dof == 1:
			xyz=self.raw_torch.narrow(1,15,3).view(-1,3)
			quat=self.raw_torch.narrow(1,18,4).view(-1,4)
		else:
			xyz=self.raw_torch.narrow(1,25,3).view(-1,3)
			quat=self.raw_torch.narrow(1,28,4).view(-1,4)
		poses = torch.cat((xyz,quat), 1)
		return poses

	def get_angles(self):
		if self.n_dof == 1:
			angles=self.raw_torch.narrow(1,22,1).view(-1,1)
		else:
			angles = self.raw_torch.narrow(1,32,2).view(-1,2)
		return angles

	def compute_bounds(self):
		self.bounds=np.zeros((self.raw_labels.shape[1],2))
		for d in range(self.raw_labels.shape[1]):
			dmin=min(self.raw_labels[:,d])
			dmax=max(self.raw_labels[:,d])
			self.bounds[d,0]=dmin
			self.bounds[d,1]=dmax

	def normalize_dim(self, data, bounds, dimension):
		dmax, dmin = bounds[dimension,1], bounds[dimension,0]
		data[:,dimension] = (data[:,dimension]-dmin)/(dmax-dmin)

	def normalize_labels(self):
		for d in range(self.raw_labels.shape[1]):
			dmin = self.bounds[d,0]
			dmax = self.bounds[d,1]
			if (dmin==0 and dmax==0) or (dmin==1 and dmax==1) or (dmin==dmax):
				pass
			else:
				self.normalize_dim(self.raw_labels, self.bounds, d)

# %%
if __name__ == '__main__':
	###### TEST THAT SHIT
	from magic.noise_models import DropPixels, DropPixelsMasked
	from magic.shape_inference.utils import interpret_labels, convert_dict_to_real, convert_to_real, expand_labels
	import matplotlib.pyplot as plt


	# d = MixtureDataset(0,
	# 				   # '../data/back-to-the-past/drawer',
	# 				   '../data/524test2/microwave',
	# 				   # '../data/fake-toaster/drawer/',
	# 				   bounds=[],
	# 				   n_dof=1,
	# 				   normalize=False,
	# 				   transform=None,
	# 				   test=False,
	# 				   keep_columns=np.arange(22),
	# 				   one_columns=[],
	# 				   preserve_labels=False,
	# 				   distractor_prob=0.0)


	data = MixtureDataset(160,
						'/Users/abba/projects/magic/magic/data/fake-toaster/cabinet2',
						# '/Users/abba/projects/magic/magic/data/shape-all-copy/',
						n_dof=2,
						normalize=True,
						transform=DropPixels(p=0.1))

	loader = torch.utils.data.DataLoader(data, batch_size=1,
	                                        shuffle=True, num_workers=0,
	                                        pin_memory=True)

	# for i,X in enumerate(loader):
	# 	img = X['depth']
	# 	plt.imshow(img[0,0] * 12.0)
	# 	plt.show()
	# 	break
	# # indices = np.arange(0,160,16)
	# print('axes \n', data.axes[indices, :])
	# print('radii \n', data.radii[indices, :])
	# print('configs \n', data.configs[indices, :])
	# print('geoms \n', data.geoms[indices, :])
	# print('poses \n', data.poses[indices, :])

	# labels = torch.cat((data.axes,data.configs,data.radii,data.geoms,data.poses), dim=1)
	# labels = data.full_labels
	# x = labels
	# zero_columns = [i for i in range(x.shape[1]) if not x[:,i].byte().any()]
	# one_columns = [i for i in range(x.shape[1]) if x[:,i].byte().all()]
	# keep_columns = [i for i in range(x.shape[1]) if (i not in zero_columns) and (i not in one_columns)]
	# filtered_x = x[:, keep_columns]
	#
	# restored_labels = expand_labels(data.full_labels, data.labels, data.keep_columns, data.one_columns)
	# print((restored_labels - labels).sum())
	#

	#  visualize noised data
	# plt.figure(figsize=(6,8))
	# plt.imshow(data[0]['depth'][0].numpy())
	# plt.show()

	#  inspect label histogram
	# print(data.axes.shape)
	# plt.subplot(131); plt.hist(data.axes[:,0])
	# plt.subplot(132); plt.hist(data.axes[:,1])
	# plt.subplot(133); plt.hist(data.axes[:,2])
	# plt.show()
	# print(data.configs.shape)
	# plt.hist(data.configs.numpy() * 180 / 3.14159)
	# plt.show()



#%%

# Idea for removing constant zero entries:
# two functions: collapse and expand labels
# the first will remove all columns of labels which are consistently empty
# the second will add those columns back such that the label is interpretable again
