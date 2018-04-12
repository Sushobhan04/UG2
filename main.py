import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
from UG2.models.srnet import SRNet, feat_ext, Classifier, vgg16_classifier
from collections import OrderedDict
import copy
import json

def save_model(model, optimizer, path = "/", filename = 'check_point.pth'):
	torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, os.path.join(path, filename))

def load_model(model, path, name, mode = "parallel"):
	state_dict = torch.load(os.path.join(path, name))["model"]
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = ""
		if mode == "single" and k.startswith("module."):
			name = k[7:]

		elif mode == "parallel" and not k.startswith("module."):
			name = "module."+k

		else:
			name = k

		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)

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
	stop_training_flag = False

	if config.discriminator == "feat_ext":
		discriminator = feat_ext()
	elif config.discriminator == "classifier":
		discriminator = Classifier((224, 224), vgg16_classifier(), mapping_list = config.mapping_list)

	if config.cuda:
		model.cuda()
		discriminator.cuda()

	if config.data_parallel:
		model = nn.DataParallel(model)
		discriminator = nn.DataParallel(discriminator)

	if config.resume_training_flag:
		if config.data_parallel:        
			load_model(model, config.resume_model_path, config.resume_model_name, mode = "parallel")
		else:        
			load_model(model, config.resume_model_path, config.resume_model_name, mode = "single")

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = config.lr_step_list, gamma= config.step_decay_param)

	loss_fn = nn.MSELoss().cuda()
	loss_history = []

	for i in range(config.epochs):

		loss_arr = []

		if stop_training_flag:
			break

		for data_file in config.data_files:

			dataset = data_utils.DatasetFromFile(config.data_path, data_file, config.img_size, data_format = config.data_format)
			try:
				data_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers)
				start = time.time()

				for batch in data_loader:

					x = data_utils.convert_to_torch_tensor(batch["data"], from_numpy = False, cuda = config.cuda)
					y = data_utils.convert_to_torch_tensor(batch["label"], from_numpy = False, cuda = config.cuda)
					optimizer.zero_grad()

					y_pred = model(x)

					if config.discriminator == "feat_ext":
						loss = loss_fn(discriminator(y_pred), discriminator(y))
					elif config.discriminator == "classifier":
						loss = loss_fn(discriminator(y_pred), y)

					loss_arr.append(loss.data[0])

					loss.backward()

			except KeyboardInterrupt:
				stop_training_flag = True

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

	if config.data_parallel:
		load_model(model, config.model_path, config.model_name, mode = "parallel")
	else:
		load_model(model, config.model_path, config.model_name, mode = "single")

	_img = data_utils.convert_to_torch_tensor(img)

	out = model(_img)
	out = out.data.cpu().numpy()

	if config.hist_eq:
		out = image_utils.hist_match(out[0], img[0])

	return out

def test_batch(inp, config):
	model = SRNet()

	if config.data_parallel:
		model = nn.DataParallel(model)

	if config.cuda:
		model.cuda()

	if config.data_parallel:
		load_model(model, config.model_path, config.model_name, mode = "parallel")
	else:
		load_model(model, config.model_path, config.model_name, mode = "single")

	_inp = copy.deepcopy(inp)

	_inp = data_utils.convert_to_torch_tensor(_inp)

	out = model(_inp)
	out = out.data.cpu().numpy()

	if config.hist_eq:
		out_arr = []
		for img in out:
			out_arr.append(image_utils.hist_match(out[0], img[0]))

	return out

def train_imagenet_classifier(config):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	stop_training_flag = False
	
	model = SRNet()
	classifier = Classifier((224, 224), vgg16_classifier())

	if config.data_parallel:
		model = nn.DataParallel(model)
		classifier = nn.DataParallel(classifier)

	if config.cuda:
		model.cuda()
		classifier.cuda()

	if config.resume_training_flag:
		load_model(model, config.resume_model_path, config.resume_model_name)

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	loss_fn = nn.MSELoss().cuda()
	loss_history = []

	for i in range(config.epochs):

		loss_arr = []

		if stop_training_flag:
			break

		try:

			train_loader = DataLoader(
				datasets.ImageFolder(config.train_dir, transforms.Compose([
					transforms.Scale(256),
					transforms.ToTensor(),
					normalize])))

			start = time.time()
			
			for batch in data_loader:

				x = data_utils.convert_to_torch_variable(batch["data"], from_numpy = False)
				y = data_utils.convert_to_torch_variable(batch["label"], from_numpy = False)
				optimizer.zero_grad()

				y_pred = model(x)

				if config.discriminator == "feat_ext":
					loss = loss_fn(discriminator(y_pred), discriminator(y))
				elif config.discriminator == "classifier":
					loss = loss_fn(discriminator(y_pred), y)

				loss_arr.append(loss.data[0])

				loss.backward()
				optimizer.step()

		except KeyboardInterrupt:
			stop_training_flag = True

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


def train_ug2_classifier(config):
	
	model = SRNet()
	stop_training_flag = False

	discriminator = Classifier(vgg16_classifier(), (224, 224))

	if config.cuda:
		model.cuda()
		discriminator.cuda()

	if config.data_parallel:
		model = nn.DataParallel(model)
		discriminator = nn.DataParallel(discriminator)

	if config.resume_training_flag:
		load_model(model, config.resume_model_path, config.resume_model_name)

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = config.lr_step_list, gamma= config.step_decay_param)
	loss_fn = nn.MSELoss().cuda()
	loss_history = []
	epoch_loss_arr = []

	for i in range(config.epochs):

		loss_arr = []

		if stop_training_flag:
			break

		start = time.time()

		for data_file in config.data_files:

			dataset = data_utils.DatasetFromFile(config.data_path, data_file, config.img_size, data_format = config.data_format)
			try:
				data_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers)

				for batch in data_loader:

					x = data_utils.convert_to_torch_tensor(batch["data"], from_numpy = False, cuda = config.cuda)
					y = data_utils.convert_to_torch_tensor(batch["label"], from_numpy = False, cuda = config.cuda)
					optimizer.zero_grad()

					y_pred = model(x)
					y_pred = discriminator(y_pred)

					if config.mapping_list is not None:
						mapped_output = []

						for j in range(len(config.mapping_list)):
							mapping = data_utils.convert_to_torch_tensor(np.array(config.mapping_list[j]), from_numpy = True, cuda = False, dtype = "int64")
							group = torch.index_select(y_pred.cpu(), 1, mapping)
							mapped_output.append(torch.sum(group))

						y_pred = torch.stack(mapped_output, dim = 1)
					
					loss = loss_fn(y_pred, y.cpu())

					loss_arr.append(loss.data[0])

					loss.backward()

			except KeyboardInterrupt:
				stop_training_flag = True

		optimizer.step()

		mean_epoch_loss = np.mean(loss_arr)
		epoch_loss_arr.append(mean_epoch_loss)
		if i%config.print_step == 0:
			print("time: ", time.time() - start, " Error: ", mean_epoch_loss)

		if i%config.checkpoint == 0:
			save_model(model, optimizer, path = config.model_path, filename = config.model_name + "_chkpt_" + str(i) + ".pth")
			print("saved checkpoint at epoch: ", i)

		if exit_training(mean_epoch_loss, loss_history, window = config.exit_loss_window, epsilon = config.loss_epsilon):
			break

	save_model(model, optimizer, path = config.model_path, filename = config.model_name)
	print("Model saved as: ", config.model_name)

	with open(os.path.join(config.model_path, config.model_name + ".txt"), 'w') as outfile:
    	json.dump(epoch_loss_arr, outfile)
