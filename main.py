import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np
import time
import h5py
import os
from UG2.utils import data as data_utils
from UG2.utils import image as image_utils
from UG2.models.srnet import SRNet, feat_ext

def save_model(model, optimizer, path = "/", filename = 'check_point.pth'):
	torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, os.path.join(path, filename))

def load_model(model, path, name):
	loaded_model = torch.load(os.path.join(path, name))["model"]
	model.load_state_dict(loaded_model)


def train(config):
	model = SRNet()
	feature_extractor = feat_ext()

	if config.data_parallel:
		model = nn.DataParallel(model)
		feature_extractor = nn.DataParallel(feature_extractor)

	if config.cuda:
		model.cuda()
		feature_extractor.cuda()

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	loss_fn = nn.MSELoss().cuda()

	for i in range(config.epochs):
		loss_arr = []

		batch_generator = data_utils.BatchGenerator(config.train_files, config.batch_size)

		start = time.time()
		for x, y in batch_generator:
			optimizer.zero_grad()

			y_pred = model(x)

			loss = loss_fn(feature_extractor(y_pred), feature_extractor(y))
			loss_arr.append(loss.data[0])

			loss.backward()
			optimizer.step()

		if i%config.print_step == 0:
			print("time: ", time.time() - start, " Error: ", np.mean(loss_arr))

		if i%config.checkpoint == 0:
			save_model(model, optimizer, path = config.model_save_path)
			print("saved checkpoint at epoch: ", i)


	save_model(model, optimizer, path = config.model_save_path, filename = config.model_name)
	print("Model saved as: ", config.model_name)

def test(model, img, hist_eq = True):
	_img = data_utils.convert_to_torch_variable(img)

	out = model(_img)
	out = out.data.cpu().numpy()

	if hist_eq:
		out = image_utils.hist_match(out, img)

	return out





