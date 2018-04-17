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
from UG2.models.srnet import SRNet, feat_ext, Classifier, pretrained_classifier, weights_init
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

	if config.weights_init:
		model.apply(weights_init)

	stop_training_flag = False

	feat_extractor = feat_ext(config.ext_type, cuda = config.cuda)
	
	if config.discriminator == "classifier":
		classifier = Classifier(pretrained_classifier(config.classifier_type, cuda = config.cuda), (224, 224))

	if config.cuda:
		model.cuda()
		feat_extractor.cuda()

		if config.discriminator == "classifier":
			classifier.cuda()

	if config.data_parallel:
		model = nn.DataParallel(model)
		feat_extractor = nn.DataParallel(feat_extractor)

		if config.discriminator == "classifier":
			classifier = nn.DataParallel(classifier)

	if config.resume_training_flag:
		if config.data_parallel:        
			load_model(model, config.resume_model_path, config.resume_model_name, mode = "parallel")
		else:        
			load_model(model, config.resume_model_path, config.resume_model_name, mode = "single")

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.lr)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = config.lr_step_list, gamma= config.step_decay_param)

	loss_fn = nn.MSELoss()

	if config.cuda:
		loss_fn.cuda()

	loss_history = []
	epoch_loss_arr = []


	for i in range(config.epochs):

		loss_arr = []
		start = time.time()


		if stop_training_flag:
			break

		for data_file in config.data_files:

			dataset = data_utils.DatasetFromFile(config.data_path, data_file, config.img_size, data_format = config.data_format)
			try:
				data_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers)

				for batch in data_loader:

					x = data_utils.convert_to_torch_tensor(batch["data"], from_numpy = False, cuda = config.cuda)
					y = data_utils.convert_to_torch_tensor(batch["label"], from_numpy = False, cuda = config.cuda)
					
					if config.discriminator == "classifier":
						cl = data_utils.convert_to_torch_tensor(batch["class"], from_numpy = False, cuda = config.cuda)
					optimizer.zero_grad()

					y_pred = model(x)
					y_pred_feat = feat_extractor(y_pred)

					feat_loss = loss_fn(y_pred_feat, feat_extractor(y))

					total_loss = feat_loss

					if config.discriminator == "classifier":
						y_pred_imgnet_class = classifier(y_pred)
						y_pred_class = ug2_classifier_loss(y_pred_imgnet_class.cpu(), config)
						class_loss = loss_fn(y_pred_class, cl.cpu())
						total_loss = feat_loss.cpu() + class_loss


					loss_arr.append(total_loss.data[0])

					total_loss.backward()
					optimizer.step()

			except KeyboardInterrupt:
				stop_training_flag = True

		mean_epoch_loss = np.mean(loss_arr)
		epoch_loss_arr.append(mean_epoch_loss)

		if i%config.print_step == 0:
			print("time: ", time.time() - start, " Error: ", mean_epoch_loss)

		if i%config.checkpoint == 0:
			if config.save_separate_chkpt:
				save_model(model, optimizer, path = config.model_path, filename = config.model_name + "_chkpt"+str(i)+".pth")
			else:
				save_model(model, optimizer, path = config.model_path, filename = config.model_name + "_chkpt.pth")
			print("saved checkpoint at epoch: ", i)

		if exit_training(mean_epoch_loss, loss_history, window = config.exit_loss_window, epsilon = config.loss_epsilon):
			stop_training_flag = True


	save_model(model, optimizer, path = config.model_path, filename = config.model_name)
	print("Model saved as: ", config.model_name)

	with open(os.path.join(config.model_path, config.model_name + ".txt"), 'w') as outfile:
		json.dump(epoch_loss_arr, outfile)

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
		out = image_utils.hist_match(out[0], img)

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
		for i in range(out.shape[0]):
			out_arr.append(image_utils.hist_match(out[i], inp[i]))

	return np.array(out_arr)


def ug2_classifier_loss(y_pred, config):
	mapped_output = []

	for j in range(len(config.mapping_list)):
		mapping = data_utils.convert_to_torch_tensor(np.array(config.mapping_list[j]), from_numpy = True, cuda = False, dtype = "int64")
		group = torch.index_select(y_pred, 1, mapping)
		mapped_output.append(torch.sum(group, dim = 1))

	y_pred = torch.stack(mapped_output, dim = 1)


	return y_pred
