import argparse

import numpy as np
import torch

from magic.mixture.dataset import MixtureDataset
from magic.mixture.real_dataset import RealDataset
from magic.mixture.models import KinematicMDNv3
from magic.mixture.mixture_errors import mixture_error_table

parser = argparse.ArgumentParser(description="Train object learner on articulated object dataset.")
parser.add_argument('--name', type=str, help='jobname', default='test_analysis')
parser.add_argument('--ntest', type=int, help='how many sim datapoints', default=16000)
parser.add_argument('--n_gaussians', type=int, help='number of components in mixture', default=20)
parser.add_argument('--real-dir', type=str, default='/Volumes/Passport/replay_annotations/microwave0/')
parser.add_argument('--fake-dir', type=str, default='../data/524test2/microwave')
parser.add_argument('--ndof', type=int, default=1, help='how many degrees of freedom in the object class?')
parser.add_argument('--obj', type=str, default='microwave')
parser.add_argument('-b','--bounds', type=str, default='/Users/abba/projects/magic/magic-reality/bounds/carrot-big-data-microwave-bounds.npy')
parser.add_argument('--real', action='store_true', help='give this flag if you want results on real data. else, fake.')
parser.add_argument('--model', type=str, default='nets/resnet-dropout-microwave.net')
args=parser.parse_args()

if args.real:
	if args.obj == 'microwave':
		dirs = ['/Volumes/Passport/replay_annotations/microwave0/',
				'/Volumes/Passport/replay_annotations/microwave1/',
				'/Volumes/Passport/replay_annotations/microwave2/',
				'/Volumes/Passport/replay_annotations/microwave3/']
		mask_val = 2.2
		config = 0
		args.model = 'nets/resnet-dropout-microwave.net'
		args.bounds = '../bounds/carrot-big-data-microwave-bounds.npy'
		args.fake_dir = '../data/524test2/microwave/'
		args.obj = 'microwave'
		args.ndof = 1

	if args.obj == 'microwave-open':
		# TODO: try microwave0 too, I think it will suck tho.
		dirs = ['/Volumes/Passport/replay_annotations/microwave-open0/']
		mask_val = 2.2
		config = -1.57
		args.model = 'nets/resnet-dropout-microwave.net'
		args.bounds = '../bounds/carrot-big-data-microwave-bounds.npy'
		args.fake_dir = '../data/524test2/microwave/'
		args.obj = 'microwave'
		args.ndof = 1

	elif args.obj == 'toaster':
		dirs = ['/Volumes/Passport/replay_annotations/toaster0/',
				'/Volumes/Passport/replay_annotations/toaster1/']
		mask_val = 2.5
		config = 0
		args.obj = 'toaster'
		args.ndof = 1
		args.bounds = '../bounds/small-toaster.npy'
		# args.model = 'nets/jul3-carrot-small-toaster.net'
		args.model = 'nets/jul3-carrot-small-toaster-drop8.net'
		args.fake_dir = '../data/fake-toaster/toaster'

	elif args.obj == 'refrigerator':
		dirs = ['/Volumes/Passport/replay_annotations/refrigerator0/']
		args.obj = 'refrigerator'
		args.ndof = 2
		args.fake_dir = '/Users/abba/projects/magic/magic/data/fake-toaster/refrigerator'
		args.bounds = '/Users/abba/projects/magic/magic/bounds/small-refrigerator.npy'
		args.model = 'nets/jul3-carrot-small-refrigerator.net'
		mask_val = 5
		config = 0
		# config = [0,0]

	elif args.obj == 'cabinet2':# -----cabinet2
		dirs = ['/Volumes/Passport/replay_annotations/cabinet2/']
		args.obj = 'cabinet2'
		args.ndof = 2
		args.fake_dir = '/Users/abba/projects/magic/magic/data/fake-toaster/cabinet2'
		args.bounds = '/Users/abba/projects/magic/magic/bounds/back-to-the-past-cabinet2.npy'
		args.model = 'nets/jul2-nonoise-cabinet2.net'
		mask_val = 2.5
		config = [0,0]

	elif args.obj == 'cabinet': # -----cabinet
		dirs = ['/Volumes/Passport/replay_annotations/cabinetL/',
				'/Volumes/Passport/replay_annotations/cabinetR/']
		args.obj = 'cabinet'
		args.ndof = 1
		args.fake_dir = '/Users/abba/projects/magic/magic/data/fake-toaster/cabinet'
		args.bounds = '/Users/abba/projects/magic/magic/bounds/carrot-big-data-cabinet-bounds.npy'
		args.model = 'nets/resnet-dropout-cabinet.net'
		mask_val = 2.5
		config = 0

	elif args.obj == 'drawer':
		# -----drawer
		dirs = ['/Volumes/Passport/replay_annotations/drawer0/']
		args.obj = 'drawer'
		args.ndof = 1
		args.fake_dir = '/Users/abba/projects/magic/magic/data/fake-toaster/drawer'
		# args.bounds = '/Users/abba/projects/magic/magic/bounds/back-to-the-past-drawer.npy'
		# args.model = 'nets/jul1-drawer.net'
		args.bounds = '../bounds/small-drawer.npy'
		args.model = 'nets/jul3-carrot-drawer-small.net'
		mask_val = 3.0
		config = 0
		#####################################

torch.manual_seed(0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bounds = np.load(args.bounds)

pre_dataset_fake = MixtureDataset(args.ntest,
							  args.fake_dir,
							  n_dof=args.ndof,
							  normalize=True)


dataset_fake = MixtureDataset(args.ntest,
							  args.fake_dir,
							  n_dof=args.ndof,
							  normalize=True,
							  bounds=bounds,
							  keep_columns=pre_dataset_fake.keep_columns,
							  one_columns=pre_dataset_fake.one_columns)

model = KinematicMDNv3(n_gaussians=args.n_gaussians,
					   out_features=dataset_fake.labels.shape[1],
					   p=0.0)

model.load_state_dict(torch.load(args.model, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.float().to(device).eval()

if args.real:
	ndof = 1 if (args.real and args.obj == 'refrigerator') else args.ndof
	dataset_real = RealDataset(dirs,
							   ndof,
							   config,
							   bounds,
							   dataset_fake.keep_columns,
							   dataset_fake.one_columns,
							   dataset_fake.full_labels,
							   mask_val=mask_val,
							   obj=args.obj)

	# dataset_real.N=N
	dataloader = torch.utils.data.DataLoader(dataset_real,
											 # batch_size=len(dataset_real),
											 batch_size = 1,
											 shuffle = True)
else:
	dataloader=torch.utils.data.DataLoader(dataset_fake, batch_size=128, shuffle=True)

table = mixture_error_table(args.name,
							model,
							dataloader,
							device,
							args.n_gaussians,
							args.obj,
							args.ndof,
							normalize=True,
							real=args.real,
							sample_error=False,
							pointcloud_error=True)
print(table)
