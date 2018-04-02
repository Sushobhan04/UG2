import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np
import time
import h5py
import os
import matplotlib.pyplot as plt
import random
from UG2.utils import data as data_utils
from UG2.utils import image as image_utils
from UG2.models.srnet import SRNet, feat_ext

def save_model(model, optimizer, path = "/", filename = 'check_point.pth'):
	torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, os.path.join(path, filename))

def load_model(model, path, name):
	loaded_model = torch.load(os.path.join(path, name))["model"]
	model.load_state_dict(loaded_model)

def exit_training(error, error_history, window = 5, epsilon = 0.01):
	if len(error_history)==0:
		min_error = np.inf
	else:
		min_error = np.min(error_history[-window:])
		
	if (error - min_error)> epsilon:
		return True
	else:
		return False
    
def train(config):
        
	model = SRNet()
	feature_extractor = feat_ext()

	if config.data_parallel:
		model = nn.DataParallel(model)
		feature_extractor = nn.DataParallel(feature_extractor)

	if config.cuda:
		model.cuda()
		feature_extractor.cuda()

	if config.resume_training_flag:
		load_model(model, config.resume_model_path, config.resume_model_name)

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	loss_fn = nn.MSELoss().cuda()
	loss_history = []

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

		mean_epoch_loss = np.mean(loss_arr)
		if i%config.print_step == 0:
			print("time: ", time.time() - start, " Error: ", mean_epoch_loss)

		if i%config.checkpoint == 0:
			save_model(model, optimizer, path = config.model_path)
			print("saved checkpoint at epoch: ", i)

		if exit_training(mean_epoch_loss, loss_history, window = config.exit_loss_window, epsilon = config.loss_epsilon):
			break


	save_model(model, optimizer, path = config.model_path, filename = config.model_name)
	print("Model saved as: ", config.model_name)

def test_single(img, config):

	model = SRNet()

	if config.data_parallel:
		model = nn.DataParallel(model)

	if config.cuda:
		model.cuda()

	load_model(model, config.model_path, config.model_name)

	_img = data_utils.convert_to_torch_variable(img)

	out = model(_img)
	out = out.data.cpu().numpy()

	if config.hist_eq:
		out = image_utils.hist_match(out, img)

	return out





